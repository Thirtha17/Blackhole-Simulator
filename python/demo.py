import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import bhsim

WIDTH = 400
HEIGHT = 240
FOV_DEG = 50.0
BH_ANG_DEG = 4.8
LENS_STRENGTH = 0.55
DUMMY = np.zeros((2, 2, 3), dtype=np.uint8)


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


init_pitch = 6.0
init_distance = 1.0
img = render_frame(init_pitch, init_distance)

fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20)

im_artist = ax.imshow(img)
ax.set_title("Blackhole Simulator")
# ax.axis("off")

ax_dist = plt.axes([0.12, 0.10, 0.75, 0.03])
ax_pitch = plt.axes([0.12, 0.05, 0.75, 0.03])

dist_slider = Slider(ax_dist, "Distance", 0.4, 2.0, valinit=init_distance, valstep=0.1)
pitch_slider = Slider(ax_pitch, "Pitch", -20.0, 60.0, valinit=init_pitch, valstep=5.0)


def on_change(_):
    pitch = float(pitch_slider.val)
    dist = float(dist_slider.val)
    frame = render_frame(pitch, dist)
    im_artist.set_data(frame)
    fig.canvas.draw_idle()


dist_slider.on_changed(on_change)
pitch_slider.on_changed(on_change)

plt.show()
