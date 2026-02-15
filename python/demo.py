import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import bhsim

def bloom(img_rgba, strength=1.35, threshold=120):
    img = img_rgba[..., :3].astype(np.float32)

    # bright-pass
    bright = np.clip(img - threshold, 0, 255)

    # cheap blur (repeat neighbor averaging)
    for _ in range(6):
        bright = (
            bright +
            np.roll(bright, 1, 0) + np.roll(bright, -1, 0) +
            np.roll(bright, 1, 1) + np.roll(bright, -1, 1)
        ) / 5.0

    out = np.clip(img + strength * bright, 0, 255).astype(np.uint8)
    rgba = img_rgba.copy()
    rgba[..., :3] = out
    return rgba

# Render (sky_img arg still required by signature; pass dummy)
dummy = np.zeros((2, 2, 3), dtype=np.uint8)

img = bhsim.render_sky_with_bh(
    width=1000,
    height=600,
    fov_deg=75.0,
    bh_angular_radius_deg=4.8,
    lens_strength=0.55,
    sky_img=dummy
)

# For line-art mode, keep raw output (no bloom).

plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.axis("off")
plt.show()
