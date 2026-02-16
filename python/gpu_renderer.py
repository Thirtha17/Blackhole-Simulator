"""GPU renderer (optional) using PyTorch.

This is a separate path from the C++ extension and is designed to run on:
- CUDA GPU (Linux/Windows)
- MPS GPU (Apple Silicon)
- CPU fallback (for functional testing)
"""

from __future__ import annotations

import math
from typing import Tuple


try:
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for gpu_renderer.py. "
        "Install torch first (with CUDA or MPS support)."
    ) from exc


def _pick_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def _grad_ln_n(r: torch.Tensor, M: float) -> torch.Tensor:
    rho = torch.clamp(torch.linalg.norm(r, dim=-1, keepdim=True), min=1e-8)
    q = M / (2.0 * rho)
    one_p = 1.0 + q
    one_m = torch.clamp(1.0 - q, min=1e-6)
    dq = -q / rho
    dlnn_drho = dq * (3.0 / one_p + 1.0 / one_m)
    rhat = r / rho
    return rhat * dlnn_drho


def _deriv(r: torch.Tensor, v: torch.Tensor, M: float) -> Tuple[torch.Tensor, torch.Tensor]:
    g = _grad_ln_n(r, M)
    proj = (v * g).sum(dim=-1, keepdim=True)
    dv = g - v * proj
    dr = v
    return dr, dv


def _rk4_step(r: torch.Tensor, v: torch.Tensor, h: torch.Tensor, M: float) -> Tuple[torch.Tensor, torch.Tensor]:
    h3 = h.unsqueeze(-1)

    k1r, k1v = _deriv(r, v, M)

    r2 = r + 0.5 * h3 * k1r
    v2 = _normalize(v + 0.5 * h3 * k1v)
    k2r, k2v = _deriv(r2, v2, M)

    r3 = r + 0.5 * h3 * k2r
    v3 = _normalize(v + 0.5 * h3 * k2v)
    k3r, k3v = _deriv(r3, v3, M)

    r4 = r + h3 * k3r
    v4 = _normalize(v + h3 * k3v)
    k4r, k4v = _deriv(r4, v4, M)

    r_next = r + (h3 / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
    v_next = _normalize(v + (h3 / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v))
    return r_next, v_next


def render_sky_with_bh_gpu(
    width: int = 640,
    height: int = 360,
    fov_deg: float = 50.0,
    bh_angular_radius_deg: float = 4.8,
    lens_strength: float = 0.55,
    cam_yaw_deg: float = 0.0,
    cam_pitch_deg: float = 6.0,
    cam_distance_scale: float = 1.0,
    max_steps: int = 1100,
    device: str | None = None,
) -> torch.Tensor:
    """Return RGBA uint8 image as torch tensor on CPU (H, W, 4)."""

    dev = _pick_device(device)
    dtype = torch.float32

    fov = math.radians(fov_deg)
    aspect = float(width) / float(height)
    warp = max(0.0, min(1.0, lens_strength))

    M = 0.8 + 1.4 * warp
    rho_horizon = 0.5 * M
    rho_capture = 1.03 * rho_horizon
    rho_escape = 70.0 * M

    disk_inner = 5.5 * M
    disk_outer = 20.0 * M
    disk_half_thickness = 0.08 * M

    cam_r = 27.0 * M * max(0.25, min(4.0, cam_distance_scale))
    yaw = math.radians(cam_yaw_deg)
    pitch = math.radians(cam_pitch_deg)
    cam = torch.tensor(
        [
            cam_r * math.cos(pitch) * math.sin(yaw),
            cam_r * math.sin(pitch),
            cam_r * math.cos(pitch) * math.cos(yaw),
        ],
        device=dev,
        dtype=dtype,
    )
    look_at = torch.tensor([0.0, 0.0, 0.0], device=dev, dtype=dtype)
    world_up = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=dtype)

    forward = _normalize((look_at - cam).view(1, 3))[0]
    right = _normalize(torch.cross(forward, world_up, dim=0).view(1, 3))[0]
    up = _normalize(torch.cross(right, forward, dim=0).view(1, 3))[0]

    ys = torch.linspace(1.0, -1.0, height, device=dev, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=dev, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    tan_half = math.tan(0.5 * fov)
    dirv = (
        forward.view(1, 1, 3)
        + right.view(1, 1, 3) * (gx * aspect * tan_half).unsqueeze(-1)
        + up.view(1, 1, 3) * (gy * tan_half).unsqueeze(-1)
    )
    dirv = _normalize(dirv)

    n = width * height
    r = cam.view(1, 3).repeat(n, 1)
    v = dirv.reshape(n, 3)

    rr = torch.zeros(n, device=dev, dtype=dtype)
    gg = torch.zeros(n, device=dev, dtype=dtype)
    bb = torch.zeros(n, device=dev, dtype=dtype)
    rho_min = torch.full((n,), 1e30, device=dev, dtype=dtype)
    disk_hits = torch.zeros(n, device=dev, dtype=torch.int32)
    was_in_slab = torch.zeros(n, device=dev, dtype=torch.bool)
    zero3 = torch.zeros((n, 3), device=dev, dtype=dtype)

    h_base = 0.030 * M

    for _ in range(max_steps):
        rho = torch.linalg.norm(r, dim=-1)
        alive = (rho >= rho_capture) & (rho <= rho_escape) & (disk_hits < 3)
        if not bool(alive.any()):
            break

        rho_min = torch.minimum(rho_min, rho)

        capture_no_emit = (rho < rho_capture) & (disk_hits == 0)
        rr = torch.where(capture_no_emit, torch.zeros_like(rr), rr)
        gg = torch.where(capture_no_emit, torch.zeros_like(gg), gg)
        bb = torch.where(capture_no_emit, torch.zeros_like(bb), bb)

        h = torch.full((n,), h_base, device=dev, dtype=dtype)
        h = torch.where(rho > 10.0 * M, h * 1.8, h)
        h = torch.where(rho > 18.0 * M, h * 2.8, h)
        h = torch.where(rho > 35.0 * M, h * 4.2, h)
        h = torch.where(alive, h, torch.zeros_like(h))

        prev_r = r
        r_next, v_next = _rk4_step(r, v, h, M)

        y1 = prev_r[:, 1]
        y2 = r_next[:, 1]
        slab_possible = (torch.abs(y1) < disk_half_thickness) | (torch.abs(y2) < disk_half_thickness) | ((y1 * y2) < 0.0)

        denom = y2 - y1
        t = torch.where(torch.abs(denom) > 1e-12, (-y1) / denom, torch.zeros_like(y1))
        t = torch.clamp(t, 0.0, 1.0)
        p = prev_r + (r_next - prev_r) * t.unsqueeze(-1)
        in_slab_now = torch.abs(p[:, 1]) < disk_half_thickness

        r_disk = torch.sqrt(p[:, 0] * p[:, 0] + p[:, 2] * p[:, 2])
        first_cross = in_slab_now & (~was_in_slab)
        hit_mask = alive & slab_possible & first_cross & (disk_hits < 3) & (r_disk > disk_inner) & (r_disk < disk_outer)

        if bool(hit_mask.any()):
            x = p[:, 0]
            z = p[:, 2]
            vphi = _normalize(torch.stack([-z, torch.zeros_like(z), x], dim=-1))
            to_cam = _normalize(cam.view(1, 3) - p)

            beta = torch.sqrt(torch.clamp(M / torch.clamp(r_disk, min=1e-6), min=0.0))
            beta = torch.clamp(beta, max=0.58)
            gamma = 1.0 / torch.sqrt(torch.clamp(1.0 - beta * beta, min=1e-8))
            mu = (vphi * to_cam).sum(dim=-1)
            doppler = 1.0 / (gamma * torch.clamp(1.0 - beta * mu, min=1e-4))

            ggrav = torch.sqrt(torch.clamp(1.0 - (2.0 * M) / torch.clamp(r_disk, min=2.05 * M), min=1e-4))
            emiss = torch.pow(disk_inner / r_disk, 2.15)
            I = 8.5 * emiss * torch.pow(doppler * ggrav, 3.0)

            t = torch.clamp((r_disk - disk_inner) / (disk_outer - disk_inner), min=0.0, max=1.0)
            R = 170.0 - 150.0 * t
            G = 220.0 - 160.0 * t
            B = 255.0 - 110.0 * t

            mult = torch.where(disk_hits == 0, torch.ones_like(rr), torch.full_like(rr, 1.45))
            add = hit_mask.float() * mult * I
            rr = rr + R * add
            gg = gg + G * add
            bb = bb + B * add
            disk_hits = disk_hits + hit_mask.to(torch.int32)

        was_in_slab = torch.where(
            alive & slab_possible,
            in_slab_now,
            torch.zeros_like(was_in_slab),
        )
        r = torch.where(alive.unsqueeze(-1), r_next, r)
        v = torch.where(alive.unsqueeze(-1), v_next, v)

    d = (rho_min - 2.9 * M) / max(0.45 * M, 1e-6)
    glow = torch.exp(-(d * d))
    rr = rr + 8.0 * glow
    gg = gg + 14.0 * glow
    bb = bb + 26.0 * glow

    rgb = torch.stack([rr, gg, bb], dim=-1)
    rgb = torch.clamp(rgb, 0.0, 255.0).to(torch.uint8)
    alpha = torch.full((n, 1), 255, device=dev, dtype=torch.uint8)
    out = torch.cat([rgb, alpha], dim=-1).reshape(height, width, 4)

    return out.cpu()
