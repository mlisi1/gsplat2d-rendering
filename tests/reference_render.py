# SPDX-License-Identifier: Apache-2.0
"""Brute-force surfel renderer: the executable spec the CUDA rasterizer is
tested against. Every pixel is tested against every splat, with no tiles,
bounding boxes or culling shortcuts, so any splat the kernel's binning
wrongly drops shows up as a mismatch.

The math is documented in docs/rasterizer.md. Float64 throughout,
from float32 (or fp16-rounded) inputs, so the kernel's own rounding is the
only source of difference. Small sizes only (O(pixels x splats) memory).
"""
from __future__ import annotations

import torch

from gsplat2d_rendering.sh import eval_sh


def _quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def reference_render(
    means, quats, log_scales, opacity_logits, features_dc, features_rest, sh_degree, *,
    w2c, fx, fy, cx, cy, width, height, campos, near_z=0.01,
    lowpass=True, aux=False, background=(0.0, 0.0, 0.0),
    alpha_min=1.0 / 255.0, alpha_max=0.99, t_min=1e-4,
):
    """Same arguments as gsplat2d_rendering._raster.rasterize (minus the
    performance-only flags). Returns a dict of float64 CPU tensors:
    rgb [3,H,W], depth, alpha [H,W], and with aux normal [3,H,W],
    median_depth [H,W]."""
    f64 = lambda t: torch.as_tensor(t).detach().cpu().double()
    means, quats, log_scales = f64(means), f64(quats), f64(log_scales)
    opacity_logits = f64(opacity_logits).reshape(-1)
    features_dc, features_rest = f64(features_dc), f64(features_rest)
    w2c, campos = f64(w2c), f64(campos)
    bg = torch.tensor(background, dtype=torch.float64)

    rot, trans = w2c[:, :3], w2c[:, 3]
    p = means @ rot.T + trans
    rc = rot @ _quat_to_rotmat(quats)
    t_u, t_v = rc[:, :, 0], rc[:, :, 1]
    n = torch.linalg.cross(t_u, t_v)
    s = log_scales.exp()
    o = torch.sigmoid(opacity_logits)

    c = (p * n).sum(-1)
    flip = c > 0
    n = torch.where(flip[:, None], -n, n)
    c = torch.where(flip, -c, c)
    a_u, a_v = t_u / s[:, :1], t_v / s[:, 1:2]
    b_u, b_v = (p * a_u).sum(-1), (p * a_v).sum(-1)

    if sh_degree < 0:
        rgb = features_dc[:, 0, :]
    else:
        dirs = means - campos
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)
        k = (sh_degree + 1) ** 2
        shs = torch.cat([features_dc, features_rest], dim=1)[:, :k, :]
        rgb = (eval_sh(sh_degree, shs.transpose(1, 2), dirs) + 0.5).clamp_min(0.0)

    centre = torch.stack([fx * p[:, 0] / p[:, 2] + cx, fy * p[:, 1] / p[:, 2] + cy], -1)

    keep = (o >= alpha_min) & (p[:, 2] > near_z)
    idx = torch.nonzero(keep, as_tuple=True)[0]
    # Stable sort on the float32 depth: the kernel sorts float32 bits with a
    # stable radix sort, so ties keep splat-index order.
    order = idx[torch.sort(p[idx, 2].float(), stable=True).indices]

    ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.float64),
                            torch.arange(width, dtype=torch.float64), indexing="ij")
    px, py = xs.reshape(-1) + 0.5, ys.reshape(-1) + 0.5
    d = torch.stack([(px - cx) / fx, (py - cy) / fy, torch.ones_like(px)], -1)

    npix = d.shape[0]
    T = torch.ones(npix, dtype=torch.float64)
    done = torch.zeros(npix, dtype=torch.bool)
    C = torch.zeros(npix, 3, dtype=torch.float64)
    Dz = torch.zeros(npix, dtype=torch.float64)
    N = torch.zeros(npix, 3, dtype=torch.float64)
    zmed = torch.zeros(npix, dtype=torch.float64)

    for j in order.tolist():
        den = d @ n[j]
        hit = den.abs() >= 1e-12
        tau = c[j] / torch.where(hit, den, torch.ones_like(den))
        hit &= tau > near_z
        u = tau * (d @ a_u[j]) - b_u[j]
        v = tau * (d @ a_v[j]) - b_v[j]
        rho = torch.where(hit, u * u + v * v, torch.full_like(u, float("inf")))
        if lowpass:
            # Where the screen-space blob wins (or the ray misses the
            # plane), the splat acts as a blob at its centre's depth; the
            # plane hit there can be arbitrarily far for a near edge-on disc.
            rho_px = 2.0 * ((px - centre[j, 0]) ** 2 + (py - centre[j, 1]) ** 2)
            use_px = rho_px < rho
            rho = torch.where(use_px, rho_px, rho)
            tau = torch.where(use_px, p[j, 2], tau)
        ok = ~done & torch.isfinite(rho)
        alpha = torch.clamp(o[j] * torch.exp(-0.5 * torch.where(ok, rho, torch.zeros_like(rho))),
                            max=alpha_max)
        ok &= alpha >= alpha_min
        t_next = T * (1.0 - alpha)
        stop = ok & (t_next < t_min)
        done |= stop
        ok &= ~stop
        w = torch.where(ok, alpha * T, torch.zeros_like(T))
        C += w[:, None] * rgb[j]
        Dz += w * tau
        if aux:
            N += w[:, None] * n[j]
            zmed = torch.where(ok & (T > 0.5), tau, zmed)
        T = torch.where(ok, t_next, T)

    a = 1.0 - T
    out = {
        "rgb": (C + T[:, None] * bg).T.reshape(3, height, width),
        "alpha": a.reshape(height, width),
        "depth": torch.where(a > 0, Dz / torch.where(a > 0, a, torch.ones_like(a)),
                             torch.zeros_like(a)).reshape(height, width),
    }
    if aux:
        out["normal"] = N.T.reshape(3, height, width)
        out["median_depth"] = zmed.reshape(height, width)
    return out


def look_at_w2c(eye, target, up=(0.0, 0.0, 1.0)):
    """[3, 4] world-to-camera for an OpenCV-axes camera at `eye` facing
    `target`; test scenes only."""
    eye = torch.tensor(eye, dtype=torch.float64)
    f = torch.tensor(target, dtype=torch.float64) - eye
    f = f / f.norm()
    r = torch.linalg.cross(f, torch.tensor(up, dtype=torch.float64))
    r = r / r.norm()
    dn = torch.linalg.cross(f, r)
    rot = torch.stack([r, dn, f], 0)
    return torch.cat([rot, (-rot @ eye)[:, None]], 1)


def random_scene(n, sh_degree=1, seed=0, spread=1.0, scale_range=(-4.0, -1.5)):
    """Splats scattered around the origin, for a camera looking at it from
    about 3 units away."""
    g = torch.Generator().manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * 2 * spread
    quats = torch.randn(n, 4, generator=g)
    lo, hi = scale_range
    log_scales = lo + (hi - lo) * torch.rand(n, 2, generator=g)
    logits = torch.randn(n, generator=g) * 2.0
    k = max((sh_degree + 1) ** 2 - 1, 0)
    dc = torch.randn(n, 1, 3, generator=g) * 0.8
    rest = torch.randn(n, k, 3, generator=g) * 0.2
    return means, quats, log_scales, logits, dc, rest

