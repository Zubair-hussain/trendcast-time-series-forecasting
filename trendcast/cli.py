"""Command-line interface: train, forecast, and render a WebM demo."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .dataset import generate_dataset, save_csv, load_dataset
from .model import TrendCast


def cmd_gen(args):
    data = generate_dataset(n_series=args.series, length=args.length, seed=args.seed)
    save_csv(data, args.out)
    print(f"wrote {len(data)} series -> {args.out}")


def cmd_forecast(args):
    series = load_dataset(args.data)
    if args.series_id not in series:
        sys.exit(f"series '{args.series_id}' not found")
    hist = series[args.series_id]
    model = TrendCast().fit(hist[: -args.horizon] if args.holdout else hist)
    fc = model.forecast(args.horizon)
    print(json.dumps(fc.as_dict(), indent=2))


def cmd_webm(args):
    from .demo import render_webm
    out = render_webm(args.out, series_id=args.series_id, horizon=args.horizon)
    print(f"rendered demo -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="trendcast", description="TrendCast CLI")
    sub = p.add_subparsers(required=True)

    g = sub.add_parser("gen", help="generate the synthetic dataset")
    g.add_argument("--out", default="data/trends_dataset.csv")
    g.add_argument("--series", type=int, default=40)
    g.add_argument("--length", type=int, default=120)
    g.add_argument("--seed", type=int, default=20260101)
    g.set_defaults(func=cmd_gen)

    f = sub.add_parser("forecast", help="forecast one series")
    f.add_argument("--data", default="data/trends_dataset.csv")
    f.add_argument("--series-id", default="topic_000")
    f.add_argument("--horizon", type=int, default=7)
    f.add_argument("--holdout", action="store_true")
    f.set_defaults(func=cmd_forecast)

    w = sub.add_parser("webm", help="render the WebM demo animation")
    w.add_argument("--out", default="assets/trendcast_demo.webm")
    w.add_argument("--series-id", default="topic_000")
    w.add_argument("--horizon", type=int, default=14)
    w.set_defaults(func=cmd_webm)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
