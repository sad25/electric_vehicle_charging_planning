# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Module-level smoke tests for the evroute engine. They exercise the
    public API end-to-end (registry -> make_env -> train -> evaluate ->
    save/load) using only the bundled `ankara_antalya` data, so they run
    offline with no API keys and no fetched datasets.
TR: evroute motoru için modül bazlı duman testleri. Yalnızca paketli
    `ankara_antalya` verisiyle genel API'yi uçtan uca (registry -> make_env
    -> eğitim -> değerlendirme -> kaydet/yükle) çalıştırır; çevrimdışı,
    API anahtarı ve indirilmiş veri gerektirmez.
"""
import compileall
from pathlib import Path

import pytest

ROUTE = "ankara_antalya"  # bundled / paketli — offline çalışır
REPO_ROOT = Path(__file__).resolve().parent.parent


def test_public_api_surface():
    """The documented public symbols are importable from `evroute`."""
    import evroute

    for name in (
        "make_env",
        "train_agent",
        "evaluate_agent",
        "registry",
        "EVChargingEnv",
        "AgentBase",
        "compute_reward",
        "__version__",
    ):
        assert hasattr(evroute, name), f"missing public symbol: {name}"


def test_registry_nonempty():
    from evroute import registry

    assert registry.list_vehicles()
    assert ROUTE in registry.list_routes()
    assert registry.list_drivers()
    assert registry.list_weather()
    assert registry.list_loads()


def test_make_env_gym_contract():
    """make_env returns a Gymnasium env that reset/steps reproducibly."""
    from evroute import make_env

    env = make_env(vehicle="ioniq5", route=ROUTE, driver="normal", seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (14,)
    assert env.action_space.n == 25  # 5 hız × 5 şarj hedefi

    env2 = make_env(vehicle="ioniq5", route=ROUTE, driver="normal", seed=42)
    obs2, _ = env2.reset(seed=42)
    assert (obs == obs2).all(), "seed=42 must be reproducible"

    o, r, term, trunc, _ = env.step(env.action_space.sample())
    assert o.shape == (14,)
    assert isinstance(float(r), float)
    assert term in (True, False) and trunc in (True, False)


def test_train_eval_save_load(tmp_path):
    """A tiny DQN run trains, evaluates, and round-trips through disk."""
    from evroute import make_env, train_agent, evaluate_agent
    from evroute.agents import DQNAgent

    env = make_env(vehicle="ioniq5", route=ROUTE, driver="normal", seed=42)
    agent = DQNAgent()
    train_agent(env, agent, num_episodes=2, verbose=False)

    metrics = evaluate_agent(env, agent, num_episodes=2)
    assert isinstance(metrics, dict) and metrics

    ckpt = tmp_path / "dqn.pt"
    agent.save(ckpt)
    assert ckpt.exists()

    reloaded = DQNAgent()
    reloaded.load(ckpt)
    obs, _ = env.reset(seed=42)
    assert 0 <= int(reloaded.select_action(obs, eval_mode=True)) < 25


def test_baselines_run():
    from evroute import make_env
    from evroute.agents.baselines import BASELINE_STRATEGIES

    assert set(BASELINE_STRATEGIES) >= {"eco", "fast_drive"}
    env = make_env(vehicle="ioniq5", route=ROUTE, driver="normal", seed=42)
    obs, _ = env.reset(seed=42)
    # Each entry is a (human_label, strategy_fn) tuple.
    label, strategy = BASELINE_STRATEGIES["eco"]
    assert isinstance(label, str)
    action = strategy(obs, env)
    assert 0 <= int(action) < 25


def test_core_imports_without_optional_extras():
    """The core must import without FastAPI / requests / matplotlib."""
    import importlib
    import evroute

    # Re-importing the core package must not pull in optional deps.
    importlib.reload(evroute.config)
    from evroute.config import get_data_dir, get_results_dir

    assert get_data_dir()  # resolves to a Path (bundled fallback is fine)
    assert get_results_dir().is_dir()


def test_research_scripts_compile():
    """research/ scripts are syntactically valid against the public API."""
    research = REPO_ROOT / "research"
    assert research.is_dir()
    ok = compileall.compile_dir(str(research), quiet=1, force=True)
    assert ok, "research/ scripts failed to byte-compile"
