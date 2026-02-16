import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import bhsim

plt.style.use("dark_background")

WIDTH = 400
HEIGHT = 240
FOV_DEG = 50.0
BH_ANG_DEG = 4.8
LENS_STRENGTH = 0.55
DUMMY = np.zeros((2, 2, 3), dtype=np.uint8)
PRECOMPUTE_ALL = False


def render_frame(pitch_deg: float, distance_scale: float):
    return bhsim.render_sky_with_bh(
        width=WIDTH,
        height=HEIGHT,
        fov_deg=FOV_DEG,
        bh_angular_radius_deg=BH_ANG_DEG,
        lens_strength=LENS_STRENGTH,
        sky_img=DUMMY,
        cam_yaw_deg=0.0,
        cam_pitch_deg=pitch_deg,
        cam_distance_scale=distance_scale,
    )


def _grid_values(vmin: float, vmax: float, step: float):
    n = int(round((vmax - vmin) / step))
    return [round(vmin + i * step, 6) for i in range(n + 1)]


PITCH_VALUES = _grid_values(-20.0, 60.0, 5.0)
DIST_VALUES = _grid_values(0.4, 2.0, 0.1)
FRAME_CACHE: dict[tuple[float, float], np.ndarray] = {}


def _cache_key(pitch: float, dist: float) -> tuple[float, float]:
    return (round(pitch, 6), round(dist, 6))


def precompute_frames() -> None:
    total = len(PITCH_VALUES) * len(DIST_VALUES)
    done = 0
    for p in PITCH_VALUES:
        for d in DIST_VALUES:
            key = _cache_key(p, d)
            if key not in FRAME_CACHE:
                FRAME_CACHE[key] = render_frame(p, d)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[cache] {done}/{total}")


init_pitch = 6.0
init_distance = 1.0
if PRECOMPUTE_ALL:
    precompute_frames()

img = FRAME_CACHE.get(_cache_key(init_pitch, init_distance), render_frame(init_pitch, init_distance))

fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20)

im_artist = ax.imshow(img)
ax.set_title("Blackhole Simulator")
ax.axis("off")

ax_dist = plt.axes([0.12, 0.10, 0.75, 0.03])
ax_pitch = plt.axes([0.12, 0.05, 0.75, 0.03])

dist_slider = Slider(ax_dist, "Distance", 0.4, 2.0, valinit=init_distance, valstep=0.1)
pitch_slider = Slider(ax_pitch, "Pitch", -20.0, 60.0, valinit=init_pitch, valstep=5.0)


def on_change(_):
    pitch = float(pitch_slider.val)
    dist = float(dist_slider.val)
    key = _cache_key(pitch, dist)
    frame = FRAME_CACHE.get(key)
    if frame is None:
        frame = render_frame(pitch, dist)
        FRAME_CACHE[key] = frame
    im_artist.set_data(frame)
    fig.canvas.draw_idle()


dist_slider.on_changed(on_change)
pitch_slider.on_changed(on_change)

plt.show()
