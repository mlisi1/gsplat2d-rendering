# SPDX-License-Identifier: Apache-2.0
"""The CUDA rasterizer against the brute-force reference, in every mode."""
from __future__ import annotations

import itertools

import pytest
import torch

from reference_render import look_at_w2c, random_scene, reference_render

pytest.importorskip("gsplat2d_rendering._raster._C")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from gsplat2d_rendering._raster import rasterize  # noqa: E402

TOL = 1e-4
W, H = 64, 48
CAM = dict(fx=60.0, fy=58.0, cx=31.0, cy=25.0, width=W, height=H)


def campos_of(w2c):
    return -w2c[:, :3].T @ w2c[:, 3]


def run_both(scene, sh_degree, w2c, cam=CAM, half=False, **opts):
    means, quats, ls, logits, dc, rest = scene
    if half:
        quats, ls, logits, dc, rest = (t.half() for t in (quats, ls, logits, dc, rest))
    campos = campos_of(w2c)
    perf = {k: opts.pop(k) for k in ("exact_tiles",) if k in opts}
    ref = reference_render(means, quats.float(), ls.float(), logits.float(), dc.float(),
                           rest.float(), sh_degree, w2c=w2c, campos=campos, **cam, **opts)
    dev = [t.cuda() for t in (means, quats, ls, logits, dc, rest)]
    out = rasterize(*dev, sh_degree, w2c=w2c, campos=campos, **cam, **opts, **perf)
    return ref, out


def max_err(ref, out, aux):
    errs = {k: (getattr(out, k).double().cpu() - ref[k]).abs().max().item()
            for k in ("rgb", "depth", "alpha")}
    if aux:
        errs["normal"] = (out.normal.double().cpu() - ref["normal"]).abs().max().item()
        errs["median_depth"] = (out.median_depth.double().cpu() - ref["median_depth"]).abs().max().item()
    return errs


def assert_close(ref, out, aux, tol=TOL):
    errs = max_err(ref, out, aux)
    assert all(e <= tol for e in errs.values()), errs


W2C = look_at_w2c((0.3, -3.0, 0.8), (0.0, 0.0, 0.0))


@pytest.mark.parametrize("lowpass,aux,exact,half",
                         list(itertools.product([False, True], repeat=4)))
def test_modes_match_reference(lowpass, aux, exact, half):
    scene = random_scene(1500, sh_degree=2, seed=1)
    ref, out = run_both(scene, 2, W2C, half=half, lowpass=lowpass, aux=aux, exact_tiles=exact)
    assert_close(ref, out, aux)


@pytest.mark.parametrize("sh_degree", [-1, 0, 1, 3])
def test_sh_degrees(sh_degree):
    scene = random_scene(800, sh_degree=max(sh_degree, 0), seed=2)
    if sh_degree < 0:
        scene = scene[:4] + (torch.rand(800, 1, 3),) + (torch.zeros(800, 0, 3),)
    ref, out = run_both(scene, sh_degree, W2C, lowpass=False)
    assert_close(ref, out, False)


def test_background_and_rgb8():
    scene = random_scene(300, seed=3)
    ref, out = run_both(scene, 1, W2C, lowpass=False, background=(0.2, 0.5, 1.0))
    assert_close(ref, out, False)
    dev = [t.cuda() for t in scene]
    o8 = rasterize(*dev, 1, w2c=W2C, campos=campos_of(W2C), **CAM, lowpass=False,
                   background=(0.2, 0.5, 1.0), rgb8=True).rgb8
    expect = (out.rgb.clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8)
    assert torch.equal(o8, expect)


def test_exact_tiles_same_image_fewer_pairs():
    scene = random_scene(3000, seed=4, scale_range=(-3.0, -0.5))
    dev = [t.cuda() for t in scene]
    kw = dict(w2c=W2C, campos=campos_of(W2C), width=640, height=480, fx=600.0, fy=600.0,
              cx=320.0, cy=240.0, lowpass=False, aux=True)
    box = rasterize(*dev, 1, exact_tiles=False, **kw)
    exact = rasterize(*dev, 1, exact_tiles=True, **kw)
    for k in ("rgb", "depth", "alpha", "normal", "median_depth"):
        assert torch.equal(getattr(box, k), getattr(exact, k)), k
    assert exact.num_pairs <= box.num_pairs
    assert exact.num_pairs < box.num_pairs  # tilted discs must save something


def test_empty_and_all_culled():
    z = torch.zeros
    out = rasterize(z(0, 3).cuda(), z(0, 4).cuda(), z(0, 2).cuda(), z(0).cuda(), z(0, 1, 3).cuda(),
                    z(0, 0, 3).cuda(), 0, w2c=W2C, campos=campos_of(W2C), **CAM,
                    background=(0.1, 0.2, 0.3))
    assert out.num_pairs == 0
    assert torch.allclose(out.rgb[:, 0, 0].cpu(), torch.tensor([0.1, 0.2, 0.3]))
    assert out.alpha.abs().max().item() == 0.0


def test_rejects_three_scales():
    scene = random_scene(10)
    dev = [t.cuda() for t in scene]
    dev[2] = torch.zeros(10, 3, device="cuda")
    with pytest.raises(ValueError):
        rasterize(*dev, 1, w2c=W2C, campos=campos_of(W2C), **CAM)
