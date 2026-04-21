

from __future__ import annotations

import numpy as np


def make_plane(size: float = 24.0, y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    
    h = size * 0.5
    positions = np.array(
        [
            [-h, y, -h],
            [h, y, -h],
            [h, y, h],
            [-h, y, -h],
            [h, y, h],
            [-h, y, h],
        ],
        dtype="f4",
    )
    normals = np.tile(np.array([[0.0, 1.0, 0.0]], dtype="f4"), (6, 1))
    return positions, normals


def make_torus(
    major_r: float = 1.1,
    minor_r: float = 0.38,
    major_segments: int = 48,
    minor_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    
    positions: list[list[float]] = []
    normals: list[list[float]] = []

    def torus_point(u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        ring = major_r + minor_r * cv
        p = np.array([ring * cu, minor_r * sv, ring * su], dtype="f4")
        n = np.array([cu * cv, sv, su * cv], dtype="f4")
        n = n / np.linalg.norm(n)
        return p, n

    du = 2.0 * np.pi / major_segments
    dv = 2.0 * np.pi / minor_segments

    for i in range(major_segments):
        for j in range(minor_segments):
            u0, u1 = i * du, (i + 1) * du
            v0, v1 = j * dv, (j + 1) * dv

            p00, n00 = torus_point(u0, v0)
            p10, n10 = torus_point(u1, v0)
            p11, n11 = torus_point(u1, v1)
            p01, n01 = torus_point(u0, v1)

            for p, n in ((p00, n00), (p10, n10), (p11, n11), (p00, n00), (p11, n11), (p01, n01)):
                positions.append(p.tolist())
                normals.append(n.tolist())

    return np.array(positions, dtype="f4"), np.array(normals, dtype="f4")
