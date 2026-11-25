"""Render an animated WebM showing TrendCast forecasting a series in real time."""

from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from .dataset import generate_dataset
from .model import TrendCast

BG = "#0d0f14"
FG = "#f1efe9"
ACCENT = "#d0021b"
MUTED = "#6b7280"


def render_webm(out_path: str, series_id: str = "topic_000",
                horizon: int = 14, fps: int = 20) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    data = {s.series_id: s.values for s in generate_dataset()}
    hist = data.get(series_id, next(iter(data.values())))
    train, actual = hist[:-horizon], hist[-horizon:]

    model = TrendCast().fit(train)
    fc = model.forecast(horizon)
    x_hist = np.arange(len(train))
    x_fut = np.arange(len(train), len(train) + horizon)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("TrendCast — live forecast", color=FG, fontsize=13, pad=12, loc="left")
    ax.set_xlabel("day", color=MUTED, fontsize=9)
    ax.set_ylabel("engagement", color=MUTED, fontsize=9)

    ax.plot(x_hist, train, color=FG, lw=1.6, label="history")
    (fut_line,) = ax.plot([], [], color=ACCENT, lw=2.2, label="forecast")
    band = [None]
    badge = ax.text(0.98, 0.04, "", transform=ax.transAxes, ha="right",
                    color=BG, fontsize=10, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", fc=ACCENT, ec="none"))
    ax.legend(loc="upper left", facecolor=BG, edgecolor=MUTED,
              labelcolor=FG, fontsize=8)
    ax.set_xlim(0, len(hist) + 1)
    pad = 0.15 * (hist.max() - hist.min() + 1)
    ax.set_ylim(hist.min() - pad, hist.max() + pad)

    def init():
        fut_line.set_data([], [])
        badge.set_text("")
        return fut_line, badge

    def update(frame):
        k = frame + 1
        fut_line.set_data(x_fut[:k], fc.point[:k])
        if band[0] is not None:
            band[0].remove()
        band[0] = ax.fill_between(x_fut[:k], fc.lower[:k], fc.upper[:k],
                                  color=ACCENT, alpha=0.15)
        badge.set_text(f"{fc.label.upper()}  ·  momentum {fc.momentum:.0f}")
        return fut_line, badge, band[0]

    frames = horizon + 6
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                   frames=frames, interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libvpx-vp9",
        extra_args=["-b:v", "0", "-crf", "32", "-pix_fmt", "yuv420p"],
    )
    anim.save(out_path, writer=writer, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    render_webm("assets/trendcast_demo.webm")
