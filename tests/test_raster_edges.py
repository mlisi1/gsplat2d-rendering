# SPDX-License-Identifier: Apache-2.0
"""Hand-built splats at the places tile-based renderers break: the near
plane, image edges, edge-on and degenerate discs, alpha thresholds."""
from __future__ import annotations

import math

import pytest
import torch

from test_raster import CAM, assert_close, run_both

W, H = CAM["width"], CAM["height"]
IDENT = torch.cat([torch.eye(3, dtype=torch.float64), torch.zeros(3, 1, dtype=torch.float64)], 1)
NEAR = 0.05


def quat(axis, angle):
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    return torch.cat([torch.tensor([math.cos(angle / 2)]), axis * math.sin(angle / 2)])


def scene(splats, sh=0):
    """splats: list of (xyz, quat, (su, sv), opacity, rgb), camera at the
    origin looking down +z (w2c = identity)."""
    means = torch.tensor([s[0] for s in splats], dtype=torch.float32)
    quats = torch.stack([s[1] for s in splats])
    ls = torch.log(torch.tensor([s[2] for s in splats], dtype=torch.float32))
    o = torch.tensor([s[3] for s in splats], dtype=torch.float32)
    logits = torch.log(o / (1 - o))
    rgb = torch.tensor([s[4] for s in splats], dtype=torch.float32)
    dc = ((rgb - 0.5) / 0.28209479177387814)[:, None, :]
    return means, quats, ls, logits, dc, torch.zeros(len(splats), 0, 3)


FACING = quat((1, 0, 0), 0.0)
BG_WALL = ((0.0, 0.0, 5.0), FACING, (4.0, 4.0), 0.6, (0.2, 0.3, 0.4))


def px_to_xyz(px, py, z):
    return ((px - CAM["cx"]) / CAM["fx"] * z, (py - CAM["cy"]) / CAM["fy"] * z, z)


CASES = {
    "centre_just_past_near": [((0.01, 0.0, NEAR + 1e-3), FACING, (0.01, 0.02), 0.9, (1, 0, 0)), BG_WALL],
    "crosses_near_plane": [((0.0, 0.0, 0.3), quat((1, 0, 0), 1.2), (0.2, 0.5), 0.9, (0, 1, 0)), BG_WALL],
    "edge_on": [((0.0, 0.0, 2.0), quat((1, 0, 0), math.pi / 2), (0.5, 0.5), 0.95, (0, 0, 1)), BG_WALL],
    "fills_screen": [((0.0, 0.0, 1.0), quat((0, 1, 0), 0.3), (3.0, 3.0), 0.8, (1, 1, 0))],
    "off_left": [(px_to_xyz(-3, 20, 2.0), FACING, (0.1, 0.05), 0.9, (1, 0, 1))],
    "off_right": [(px_to_xyz(W + 3, 20, 2.0), FACING, (0.1, 0.05), 0.9, (1, 0, 1))],
    "off_top": [(px_to_xyz(30, -4, 2.0), quat((0, 0, 1), 0.7), (0.1, 0.03), 0.9, (0, 1, 1))],
    "off_bottom": [(px_to_xyz(30, H + 4, 2.0), quat((0, 0, 1), 0.7), (0.1, 0.03), 0.9, (0, 1, 1))],
    "last_row_col": [(px_to_xyz(W + 0.6, H + 0.6, 2.0), FACING, (0.03, 0.03), 0.99, (1, 1, 1))],
    "very_thin": [((0.05, 0.02, 2.0), quat((0, 0, 1), 0.4), (0.5, 1e-6), 0.9, (1, 0.5, 0)), BG_WALL],
    "alpha_min_above": [((0.0, 0.0, 2.0), FACING, (0.3, 0.3), (1 / 255) * 1.01, (1, 1, 1))],
    "alpha_min_below": [((0.0, 0.0, 2.0), FACING, (0.3, 0.3), (1 / 255) * 0.99, (1, 1, 1))],
    "equal_depth": [((0.0, 0.0, 2.0), FACING, (0.3, 0.3), 0.5, (1, 0, 0)),
                    ((0.05, 0.0, 2.0), FACING, (0.3, 0.3), 0.5, (0, 0, 1))],
    "behind_camera": [((0.0, 0.0, -1.0), FACING, (3.0, 3.0), 0.9, (1, 0, 0)), BG_WALL],
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("lowpass", [False, True])
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("near_crossing", [False, True])
def test_edge_case(name, lowpass, exact, near_crossing):
    ref, out = run_both(scene(CASES[name]), 0, IDENT, near_z=NEAR, lowpass=lowpass, aux=True,
                        exact_tiles=exact, near_crossing=near_crossing)
    assert_close(ref, out, True)


@pytest.mark.parametrize("name", ["crosses_near_plane", "fills_screen"])
def test_near_crossing_toggles_the_disc(name):
    s = scene(CASES[name][:1])
    _, off = run_both(s, 0, IDENT, near_z=NEAR, lowpass=False, near_crossing=False)
    _, on = run_both(s, 0, IDENT, near_z=NEAR, lowpass=False, near_crossing=True)
    assert off.alpha.max().item() == 0.0 and on.alpha.max().item() > 0.5


def test_off_centre_principal_point():
    cam = dict(CAM, cx=10.0, cy=40.0)
    s = scene([((0.2, -0.1, 2.0), quat((1, 1, 0), 0.5), (0.3, 0.1), 0.9, (1, 0, 0)), BG_WALL])
    ref, out = run_both(s, 0, IDENT, cam=cam, near_z=NEAR, lowpass=False, aux=True, exact_tiles=True)
    assert_close(ref, out, True)


def test_equal_depth_keeps_index_order():
    a = ((0.0, 0.0, 2.0), FACING, (0.3, 0.3), 0.5, (1, 0, 0))
    b = ((0.0, 0.0, 2.0), FACING, (0.3, 0.3), 0.5, (0, 0, 1))
    _, out_ab = run_both(scene([a, b]), 0, IDENT, near_z=NEAR, lowpass=False)
    _, out_ba = run_both(scene([b, a]), 0, IDENT, near_z=NEAR, lowpass=False)
    cy, cx = int(CAM["cy"]), int(CAM["cx"])
    assert out_ab.rgb[0, cy, cx] > out_ab.rgb[2, cy, cx]  # first-listed red wins
    assert out_ba.rgb[2, cy, cx] > out_ba.rgb[0, cy, cx]
