"""
mesh.py — RAEON Face Mesh Generator

Generates a procedural face mesh as a parametric surface.
Anime-style: V-chin silhouette + Gaussian feature bumps (eye sockets,
brow ridges, nose bridge/tip, lips, cheekbones).

Vertex groups are auto-labeled from face.json region definitions.
No external 3D model needed — face is born from math.
"""

import math
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
                u = ui / self.u_segs   # 0..1 left->right
                v = vi / self.v_segs   # 0..1 top->bottom

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

    # ── anime face shape ─────────────────────────────────────────────

    def _anime_width(self, v: float) -> float:
        """
        Anime face silhouette: width multiplier as a function of v.
        v=0 = top of head, v=1 = chin tip.
          Crown      (v 0.00-0.20): round top, 1.0 -> 0.92
          Cheekbones (v 0.20-0.45): outward swell, 0.92 -> peak ~1.0 -> 0.92
          Jaw        (v 0.45-0.72): linear taper,  0.92 -> 0.70  (CONTINUOUS at 0.45)
          Chin       (v 0.72-1.00): sharp V-point, 0.70 -> 0     (CONTINUOUS at 0.72)
        All section boundaries are C0-continuous (no width jumps).
        """
        if v < 0.20:
            t = v / 0.20
            return 1.0 - 0.08 * t * t          # 1.00 -> 0.92
        elif v < 0.45:
            t = (v - 0.20) / 0.25
            return 0.92 + 0.08 * math.sin(t * math.pi)  # 0.92 -> peak -> 0.92
        elif v < 0.72:
            t = (v - 0.45) / 0.27
            return 0.92 - 0.22 * t             # 0.92 -> 0.70
        else:
            t = (v - 0.72) / 0.28
            return 0.70 - 0.70 * (t ** 1.5)   # 0.70 -> 0 (V-chin)

    def _eye_socket_scale(self, u: float, v: float) -> float:
        """
        Radial scale factor for eye socket areas (< 1 = pulls vertex toward
        head centre along the surface normal direction).  Applied to both x
        and z so the depression follows the surface curvature instead of
        creating a flat step.
        """
        def G(u0, v0, su, sv, depth):
            du = (u - u0) / su
            dv = (v - v0) / sv
            return depth * math.exp(-0.5 * (du * du + dv * dv))

        scale = 0.0
        scale += G(0.33, 0.40, 0.155, 0.085, -0.105)   # left socket
        scale += G(0.67, 0.40, 0.155, 0.085, -0.105)   # right socket
        return 1.0 + scale   # range ≈ 0.905..1.0

    def _anime_features(self, u: float, v: float) -> float:
        """
        Z-axis Gaussian bumps (positive = protrude, negative = sink).
        Eye sockets are handled separately via _eye_socket_scale.
        """
        def G(u0, v0, su, sv, amp):
            du = (u - u0) / su
            dv = (v - v0) / sv
            return amp * math.exp(-0.5 * (du * du + dv * dv))

        z = 0.0

        # Brow ridges (shelf above eyes)
        z += G(0.33, 0.30, 0.110, 0.040, +0.048)   # left
        z += G(0.67, 0.30, 0.110, 0.040, +0.048)   # right

        # Nose bridge (thin ridge down the centre)
        z += G(0.50, 0.46, 0.030, 0.085, +0.050)

        # Nose tip (rounded button)
        z += G(0.50, 0.60, 0.050, 0.042, +0.095)

        # Nostrils (slight outward flare)
        z += G(0.43, 0.62, 0.025, 0.025, +0.030)   # left
        z += G(0.57, 0.62, 0.025, 0.025, +0.030)   # right

        # Upper lip (cupid's bow)
        z += G(0.50, 0.695, 0.105, 0.026, +0.065)

        # Lower lip (fuller)
        z += G(0.50, 0.735, 0.095, 0.032, +0.075)

        # Cheekbones (subtle outward push)
        z += G(0.22, 0.50, 0.085, 0.095, +0.040)   # left
        z += G(0.78, 0.50, 0.085, 0.095, +0.040)   # right

        return z

    # ── parametric surface ───────────────────────────────────────────

    def _parametric(self, u: float, v: float):
        """Map (u,v) -> (x,y,z) anime face surface point."""
        # u=0 left, u=1 right; theta sweeps front hemisphere only
        # v=0 top,  v=1 bottom
        #
        # Phi range: 0.10*pi..pi (18 deg to 180 deg)
        #   Top (v=0):  phi=18 deg → sin=0.31 → round crown (not a point)
        #   Mid (v=~0.44): phi=108 deg → sin=0.95 → widest zone
        #   Bottom (v=1): phi=180 deg → sin=0 → sharp V-chin
        theta = (u - 0.5) * math.pi * 0.56   # +/-~50 deg arc — flatter, no wrap artifacts
        phi   = math.pi * (0.10 + v * 0.90)  # 18..180 deg

        sin_phi   = math.sin(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Silhouette width profile
        w = self._anime_width(v)

        # Base sphere position
        x_base = self.width * w * sin_phi * sin_theta
        z_base = self.depth * sin_phi * cos_theta

        # Eye socket: radial depression (scales x AND z toward head centre)
        rs = self._eye_socket_scale(u, v)
        x = x_base * rs
        z = z_base * rs + self.depth * self._anime_features(u, v)

        y = self.height * (0.52 - v)

        return float(x), float(y), float(z)

    def _normal(self, u: float, v: float):
        """Approximate outward normal by finite difference."""
        eps = 0.001
        x0, y0, z0 = self._parametric(u, v)

        xu, yu, zu = self._parametric(min(u + eps, 1.0), v)
        xv, yv, zv = self._parametric(u, min(v + eps, 1.0))

        du = np.array([xu - x0, yu - y0, zu - z0])
        dv = np.array([xv - x0, yv - y0, zv - z0])

        n = np.cross(dv, du)   # dv x du gives outward-facing normals
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
