# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Command-line interface for the evroute engine. Sub-commands:
      list   — show registered vehicles / routes / drivers / weather / loads
      train  — train an RL agent on a scenario and save the weights
      eval   — load saved weights and evaluate on a scenario
      fetch  — pull external route/station/elevation data (needs [data])
      serve  — launch the interactive simulation server (needs [serve])
    Only the core API (make_env, train_agent, registry) is imported at
    module load; `fetch` and `serve` import their heavy/optional deps
    lazily, so `evroute list` works even without the [data]/[serve] extras.
TR: evroute motoru için komut satırı arayüzü. Alt komutlar:
      list   — kayıtlı araç / rota / sürücü / hava / yük listesi
      train  — bir senaryoda RL ajanı eğit ve ağırlıkları kaydet
      eval   — kayıtlı ağırlıkları yükle ve bir senaryoda değerlendir
      fetch  — dış rota/istasyon/yükseklik verisini çek ([data] gerekir)
      serve  — etkileşimli simülasyon sunucusunu başlat ([serve] gerekir)
    Modül yüklenirken yalnızca çekirdek API içe aktarılır; `fetch` ve
    `serve` ağır/opsiyonel bağımlılıklarını tembel yükler, böylece
    `evroute list` [data]/[serve] ekstraları olmadan da çalışır.

Usage / Kullanım:
    evroute list routes
    evroute train --agent dqn --route istanbul_ankara --episodes 5
    evroute eval  --agent dqn --model model.pt --route istanbul_ankara
    evroute fetch realdata
    evroute serve --port 8000
"""
from __future__ import annotations

import argparse
import sys


# ---------- list ----------

def _cmd_list(args: argparse.Namespace) -> int:
    from evroute import registry

    tables = {
        "vehicles": registry.list_vehicles,
        "routes": registry.list_routes,
        "drivers": registry.list_drivers,
        "weather": registry.list_weather,
        "loads": registry.list_loads,
    }
    kinds = list(tables) if args.kind == "all" else [args.kind]
    for kind in kinds:
        print(f"{kind}:")
        for key in tables[kind]():
            print(f"  {key}")
    return 0


# ---------- train ----------

def _cmd_train(args: argparse.Namespace) -> int:
    from evroute import make_env, train_agent
    from evroute.agents import DQNAgent

    if args.agent != "dqn":
        print(f"error: unknown agent '{args.agent}' (only 'dqn' supported "
              f"via the generic loop) / bilinmeyen ajan", file=sys.stderr)
        return 2

    env = make_env(vehicle=args.vehicle, route=args.route, driver=args.driver,
                   load=args.load, weather=args.weather)
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n,
                     double_dqn=not args.no_double)
    train_agent(env, agent, num_episodes=args.episodes)

    out = args.out or (f"ddqn__{args.vehicle}_{args.route}_"
                       f"{args.driver}_{args.weather}.pt")
    agent.save(out)
    print(f"saved / kaydedildi: {out}")
    return 0


# ---------- eval ----------

def _cmd_eval(args: argparse.Namespace) -> int:
    from evroute import make_env, evaluate_agent
    from evroute.agents import DQNAgent

    if args.agent != "dqn":
        print(f"error: unknown agent '{args.agent}'", file=sys.stderr)
        return 2

    env = make_env(vehicle=args.vehicle, route=args.route, driver=args.driver,
                   load=args.load, weather=args.weather)
    agent = DQNAgent(state_dim=env.observation_space.shape[0],
                     action_dim=env.action_space.n)
    agent.load(args.model)
    stats = evaluate_agent(env, agent, num_episodes=args.episodes)
    for k, v in stats.items():
        print(f"{k}: {v}")
    return 0


# ---------- fetch (lazy [data]) ----------

# which -> module under evroute.data.sources / hangi -> kaynak modülü
_FETCHERS = {
    "route": "google_routes",
    "stations": "openchargemap",
    "elevation": "google_elevation",
    "detour": "detour_vis",
    "realdata": "realdata",
}


def _cmd_fetch(args: argparse.Namespace) -> int:
    import importlib

    try:
        mod = importlib.import_module(
            f"evroute.data.sources.{_FETCHERS[args.which]}")
    except ImportError as exc:
        print(f"error: the 'fetch' command needs the [data] extra "
              f"(pip install 'evroute[data]'): {exc} / "
              f"'fetch' komutu [data] ekstrasını gerektirir",
              file=sys.stderr)
        return 1
    mod.main()
    return 0


# ---------- serve (lazy [serve]) ----------

def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # noqa: F401
    except ImportError as exc:
        print(f"error: the 'serve' command needs the [serve] extra "
              f"(pip install 'evroute[serve]'): {exc} / "
              f"'serve' komutu [serve] ekstrasını gerektirir",
              file=sys.stderr)
        return 1
    # The FastAPI app lives in evroute_serve. / Sunucu evroute_serve içindedir.
    uvicorn.run("evroute_serve.server:app", host=args.host, port=args.port)
    return 0


# ---------- parser ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evroute",
        description="EV long-distance charging-planning RL engine / "
                    "EV uzun yol şarj planlama RL motoru")
    sub = p.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("list", help="list registered entities / kayıtları listele")
    pl.add_argument("kind", nargs="?", default="all",
                    choices=["all", "vehicles", "routes", "drivers",
                             "weather", "loads"])
    pl.set_defaults(func=_cmd_list)

    def _add_env_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--vehicle", default="ioniq5")
        sp.add_argument("--route", default="istanbul_ankara")
        sp.add_argument("--driver", default="normal")
        sp.add_argument("--load", default="normal")
        sp.add_argument("--weather", default="optimal")

    pt = sub.add_parser("train", help="train an agent / ajan eğit")
    pt.add_argument("--agent", default="dqn")
    _add_env_args(pt)
    pt.add_argument("--episodes", type=int, default=2000)
    pt.add_argument("--no-double", action="store_true",
                    help="plain DQN instead of Double DQN / düz DQN")
    pt.add_argument("--out", default=None, help="output .pt path / çıktı yolu")
    pt.set_defaults(func=_cmd_train)

    pe = sub.add_parser("eval", help="evaluate saved weights / ağırlıkları değerlendir")
    pe.add_argument("--agent", default="dqn")
    pe.add_argument("--model", required=True, help="saved .pt path / model yolu")
    _add_env_args(pe)
    pe.add_argument("--episodes", type=int, default=50)
    pe.set_defaults(func=_cmd_eval)

    pf = sub.add_parser("fetch", help="fetch external data ([data]) / dış veri çek")
    pf.add_argument("which", choices=list(_FETCHERS))
    pf.set_defaults(func=_cmd_fetch)

    ps = sub.add_parser("serve", help="run the sim server ([serve]) / sunucuyu başlat")
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8000)
    ps.set_defaults(func=_cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    """EN: CLI entry point. TR: CLI giriş noktası."""
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
