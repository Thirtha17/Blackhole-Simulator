"""Optional GPU demo.

Usage:
  python python/gpu_demo.py

Requires PyTorch. Uses CUDA/MPS automatically when available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from gpu_renderer import render_sky_with_bh_gpu

plt.style.use("dark_background")


def main() -> None:
    width, height = 400, 240
    max_steps = 1100
    precompute_all = True

    def render_frame(pitch: float, distance: float):
        return render_sky_with_bh_gpu(
            width=width,
            height=height,
            fov_deg=50.0,
            bh_angular_radius_deg=4.8,
            lens_strength=0.55,
            cam_yaw_deg=0.0,
            cam_pitch_deg=pitch,
            cam_distance_scale=distance,
            max_steps=max_steps,
        ).numpy()

    def grid_values(vmin: float, vmax: float, step: float):
        n = int(round((vmax - vmin) / step))
        return [round(vmin + i * step, 6) for i in range(n + 1)]

    pitch_values = grid_values(-20.0, 60.0, 5.0)
    dist_values = grid_values(0.4, 2.0, 0.1)
    frame_cache: dict[tuple[float, float], np.ndarray] = {}

    def cache_key(pitch: float, dist: float) -> tuple[float, float]:
        return (round(pitch, 6), round(dist, 6))

    def precompute_frames() -> None:
        total = len(pitch_values) * len(dist_values)
        done = 0
        for p in pitch_values:
            for d in dist_values:
                key = cache_key(p, d)
                if key not in frame_cache:
                    frame_cache[key] = render_frame(p, d)
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"[gpu cache] {done}/{total}")

    init_pitch = 6.0
    init_distance = 1.0
    if precompute_all:
        precompute_frames()
    img = frame_cache.get(cache_key(init_pitch, init_distance), render_frame(init_pitch, init_distance))

    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20)
    im_artist = ax.imshow(img)
    ax.set_title("Blackhole Simulator (GPU)")
    ax.axis("off")

    ax_dist = plt.axes([0.12, 0.10, 0.75, 0.03])
    ax_pitch = plt.axes([0.12, 0.05, 0.75, 0.03])
    dist_slider = Slider(ax_dist, "Distance", 0.4, 2.0, valinit=init_distance, valstep=0.1)
    pitch_slider = Slider(ax_pitch, "Pitch", -20.0, 60.0, valinit=init_pitch, valstep=5.0)

    def on_change(_):
        key = cache_key(float(pitch_slider.val), float(dist_slider.val))
        frame = frame_cache.get(key)
        if frame is None:
            frame = render_frame(float(pitch_slider.val), float(dist_slider.val))
            frame_cache[key] = frame
        im_artist.set_data(frame)
        fig.canvas.draw_idle()

    dist_slider.on_changed(on_change)
    pitch_slider.on_changed(on_change)

    plt.show()


if __name__ == "__main__":
    main()
