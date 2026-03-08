"""
mesh.py — RAEON Face Mesh Generator

Generates a procedural face mesh as a parametric surface.
Vertex groups are auto-labeled from face.json region definitions.
No external 3D model needed — face is born from math.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


class FaceMesh:

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "face.json"
        with open(config_path) as f:
            self.cfg = json.load(f)

        m = self.cfg["mesh"]
        self.u_segs  = m["u_segments"]
        self.v_segs  = m["v_segments"]
        self.width   = m["width"]
        self.height  = m["height"]
        self.depth   = m["depth"]

        self.vertices  = None   # (N, 3) float32
        self.normals   = None   # (N, 3) float32
        self.uvs       = None   # (N, 2) float32
        self.indices   = None   # (M, 3) int32
        self.groups    = {}     # region_name -> [vertex_indices]

        self._build()

    # ── build ────────────────────────────────────────────────────────

    def _build(self):
        rows = self.v_segs + 1
        cols = self.u_segs + 1
        n    = rows * cols

        verts   = np.zeros((n, 3), dtype=np.float32)
        normals = np.zeros((n, 3), dtype=np.float32)
        uvs     = np.zeros((n, 2), dtype=np.float32)
        groups  = defaultdict(list)

        idx = 0
        for vi in range(rows):
            for ui in range(cols):
                u = ui / self.u_segs   # 0..1 left→right
                v = vi / self.v_segs   # 0..1 top→bottom

                x, y, z = self._parametric(u, v)
                nx, ny, nz = self._normal(u, v)

                verts[idx]   = [x, y, z]
                normals[idx] = [nx, ny, nz]
                uvs[idx]     = [u, v]

                self._label(idx, u, v, groups)
                idx += 1

        # Triangle indices (two triangles per quad)
        faces = []
        for vi in range(self.v_segs):
            for ui in range(self.u_segs):
                tl = vi * cols + ui
                tr = tl + 1
                bl = tl + cols
                br = bl + 1
                faces.extend([tl, bl, tr, tr, bl, br])

        self.vertices = verts
        self.normals  = normals
        self.uvs      = uvs
        self.indices  = np.array(faces, dtype=np.int32)
        self.groups   = dict(groups)

    def _parametric(self, u: float, v: float):
        """Map (u,v) → (x,y,z) face surface point."""
        # Horizontal: u=0 left, u=1 right, center at 0.5
        # Vertical:   v=0 top,  v=1 bottom
        theta = (u - 0.5) * np.pi * 0.90   # -pi/2..+pi/2 (front only)
        phi   = v * np.pi                   # 0..pi

        sin_phi   = np.sin(phi)
        cos_phi   = np.cos(phi)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        x = self.width  * sin_phi * sin_theta
        y = self.height * (0.52 - v)          # top=+0.52, bottom=-0.48
        z = self.depth  * sin_phi * cos_theta

        # Slight chin narrowing
        chin_factor = 1.0 - 0.3 * max(0, v - 0.75)
        x *= chin_factor

        return float(x), float(y), float(z)

    def _normal(self, u: float, v: float):
        """Approximate outward normal by finite difference."""
        eps = 0.001
        x0, y0, z0 = self._parametric(u, v)

        xu, yu, zu = self._parametric(min(u + eps, 1.0), v)
        xv, yv, zv = self._parametric(u, min(v + eps, 1.0))

        du = np.array([xu - x0, yu - y0, zu - z0])
        dv = np.array([xv - x0, yv - y0, zv - z0])

        n = np.cross(dv, du)   # dv×du gives outward-facing normals
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            return 0.0, 0.0, 1.0
        n /= ln
        return float(n[0]), float(n[1]), float(n[2])

    def _label(self, idx: int, u: float, v: float, groups: dict):
        """Assign vertex to face regions based on (u,v) position."""
        regions = self.cfg["regions"]
        for name, bounds in regions.items():
            u_min, u_max = bounds["u"]
            v_min, v_max = bounds["v"]
            if u_min <= u <= u_max and v_min <= v <= v_max:
                groups[name].append(idx)

    # ── accessors ────────────────────────────────────────────────────

    def vertex_count(self) -> int:
        return len(self.vertices)

    def group_indices(self, name: str) -> list:
        return self.groups.get(name, [])

    def group_vertices(self, name: str) -> np.ndarray:
        ids = self.group_indices(name)
        if not ids:
            return np.empty((0, 3), dtype=np.float32)
        return self.vertices[ids]

    def summary(self):
        print(f"FaceMesh  vertices={self.vertex_count()}  "
              f"triangles={len(self.indices)//3}")
        for name, ids in sorted(self.groups.items()):
            print(f"  {name:<16} {len(ids)} vertices")
