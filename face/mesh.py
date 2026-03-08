"""
mesh.py — RAEON Face Mesh Generator

Generates a procedural face mesh as a parametric surface.
Anime-style: V-chin silhouette + Gaussian feature bumps + per-vertex colours.

Per-vertex colour palette:
  (1,1,1)           — skin (shader uses u_skin_color)
  dark-brown arch   — eyebrows
  off-white circle  — sclera
  hazel circle      — iris
  near-black dot    — pupil
  bright spot       — eye highlight
  rosy rect         — lips

Vertex groups auto-labelled from face.json region definitions.
No external 3-D model — face is born from math.
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
        self.colors    = None   # (N, 3) float32  — per-vertex RGB
        self.indices   = None   # (M,)   int32
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
        colors  = np.zeros((n, 3), dtype=np.float32)
        groups  = defaultdict(list)

        idx = 0
        for vi in range(rows):
            for ui in range(cols):
                u = ui / self.u_segs   # 0..1 left->right
                v = vi / self.v_segs   # 0..1 top->bottom

                x, y, z = self._parametric(u, v)
                nx, ny, nz = self._normal(u, v)
                cr, cg, cb = self._face_colors(u, v)

                verts[idx]   = [x, y, z]
                normals[idx] = [nx, ny, nz]
                uvs[idx]     = [u, v]
                colors[idx]  = [cr, cg, cb]

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
        self.colors   = colors
        self.indices  = np.array(faces, dtype=np.int32)
        self.groups   = dict(groups)

    # ── per-vertex colour ─────────────────────────────────────────────

    def _face_colors(self, u: float, v: float) -> tuple:
        """
        Return (r, g, b) for this vertex.
        (1, 1, 1) = "use shader skin colour" (default).
        Any other value = direct base material colour (still receives lighting).

        UV→world scale at eye level: ∂x/∂u ≈ 0.57, ∂y/∂v ≈ 0.88
        → for a circle in screen-space: su = sv * 1.55

        Layer order (later = on top):
          skin → lips → sclera → iris → pupil → highlight → eyebrow
        """

        def G(u0, v0, su, sv):
            du = (u - u0) / su
            dv = (v - v0) / sv
            return math.exp(-0.5 * (du * du + dv * dv))

        def blend(base, col, alpha):
            a = min(max(alpha, 0.0), 1.0)
            return (base[0] + (col[0] - base[0]) * a,
                    base[1] + (col[1] - base[1]) * a,
                    base[2] + (col[2] - base[2]) * a)

        c = (1.0, 1.0, 1.0)   # start: pure skin

        # ── Lips ──────────────────────────────────────────────────
        # su/sv ratio ≈ 1.55 compensated
        lip_w = (G(0.50, 0.698, 0.130, 0.038) * 0.85
               + G(0.50, 0.738, 0.118, 0.034) * 0.95)
        if lip_w > 0.05:
            c = blend(c, (0.80, 0.46, 0.44), min(lip_w, 1.0))

        # ── Eyes (left u0=0.335, right u0=0.665) ──────────────────
        # All Gaussians use su ≈ sv * 1.55 for circles in screen-space
        for u0 in (0.335, 0.665):
            v0 = 0.385

            # Sclera — almond-shaped off-white
            # su=0.047 ≈ sv(0.030)*1.55 → ~circle on screen
            scl = G(u0, v0, 0.047, 0.030) * 1.5
            if scl > 0.10:
                c = blend(c, (0.93, 0.91, 0.89), min(scl, 1.0))

                # Iris — warm dark hazel circle
                ir = G(u0, v0, 0.028, 0.018) * 2.5
                c  = blend(c, (0.22, 0.13, 0.08), min(ir, 1.0))

                # Pupil — near-black dot
                pu = G(u0, v0 + 0.002, 0.016, 0.010) * 3.5
                c  = blend(c, (0.04, 0.03, 0.02), min(pu, 1.0))

                # Specular highlight — bright crescent top-left
                hi = G(u0 - 0.010, v0 - 0.010, 0.009, 0.006) * 5.0
                c  = blend(c, (0.97, 0.97, 1.00), min(hi, 1.0))

        # ── Eyebrows — dark brown arch (ON TOP of everything) ─────
        # su ≈ sv * 1.55 for consistent stroke width
        for u0, sign in ((0.326, -1.0), (0.674, +1.0)):
            bv = 0.262
            # Main arch body — thicker stroke
            bw  = G(u0, bv,              0.062, 0.016) * 2.8
            # Inner head — tapers down slightly at nose side
            bw += G(u0 - sign * 0.038, bv + 0.005, 0.022, 0.014) * 2.0
            # Outer tail — tapers thin toward temple
            bw += G(u0 + sign * 0.036, bv - 0.004, 0.022, 0.013) * 1.8

            bw = min(bw, 1.0)
            if bw > 0.12:
                c = blend(c, (0.10, 0.07, 0.04), bw)

        return c

    # ── anime face shape ──────────────────────────────────────────────

    def _anime_width(self, v: float) -> float:
        """
        Silhouette: width multiplier as fn of v (0=crown, 1=chin).
          Crown      v 0.00-0.20 : 1.00 -> 0.92
          Cheekbones v 0.20-0.45 : 0.92 -> peak ~1.00 -> 0.92  (C0)
          Jaw        v 0.45-0.72 : 0.92 -> 0.70                (C0)
          Chin       v 0.72-1.00 : 0.70 -> 0  sharp V          (C0)
        """
        if v < 0.20:
            t = v / 0.20
            return 1.0 - 0.08 * t * t
        elif v < 0.45:
            t = (v - 0.20) / 0.25
            return 0.92 + 0.08 * math.sin(t * math.pi)
        elif v < 0.72:
            t = (v - 0.45) / 0.27
            return 0.92 - 0.22 * t
        else:
            t = (v - 0.72) / 0.28
            return 0.70 - 0.70 * (t ** 1.5)

    def _eye_socket_scale(self, u: float, v: float) -> float:
        """
        Radial scale < 1 pulls the eye-socket region inward along both x and z,
        creating a smooth bowl depression that follows surface curvature.
        """
        def G(u0, v0, su, sv, d):
            du = (u - u0) / su
            dv = (v - v0) / sv
            return d * math.exp(-0.5 * (du * du + dv * dv))

        s = 0.0
        s += G(0.335, 0.390, 0.148, 0.082, -0.072)   # left  (shallower → less shadow)
        s += G(0.665, 0.390, 0.148, 0.082, -0.072)   # right
        return 1.0 + s

    def _anime_features(self, u: float, v: float) -> float:
        """
        Z-axis Gaussian bumps. Eye sockets handled by _eye_socket_scale.
        Positive = protrude, negative = sink.
        """
        def G(u0, v0, su, sv, amp):
            du = (u - u0) / su
            dv = (v - v0) / sv
            return amp * math.exp(-0.5 * (du * du + dv * dv))

        z = 0.0

        # ── Brows — sharper ridge casts a clear shadow ──────────
        z += G(0.325, 0.278, 0.060, 0.020, +0.068)   # left  brow arch
        z += G(0.675, 0.278, 0.060, 0.020, +0.068)   # right brow arch

        # ── Eyelid anatomy ──────────────────────────────────────
        # Upper lid crease shelf (overhangs eye, casts lid shadow)
        z += G(0.335, 0.352, 0.072, 0.013, +0.035)   # left
        z += G(0.665, 0.352, 0.072, 0.013, +0.035)   # right
        # Lower lid ridge (defines bottom of eye)
        z += G(0.335, 0.442, 0.062, 0.011, +0.024)   # left
        z += G(0.665, 0.442, 0.062, 0.011, +0.024)   # right

        # ── Nose ────────────────────────────────────────────────
        # Bridge — narrow vertical ridge
        z += G(0.500, 0.460, 0.028, 0.090, +0.052)
        # Tip — rounded button
        z += G(0.500, 0.605, 0.048, 0.040, +0.098)
        # Ala (nostril wings) — outward flare
        z += G(0.428, 0.618, 0.032, 0.026, +0.045)   # left
        z += G(0.572, 0.618, 0.032, 0.026, +0.045)   # right
        # Nostril base crease — subtle dip under ala
        z += G(0.435, 0.635, 0.022, 0.018, -0.018)   # left
        z += G(0.565, 0.635, 0.022, 0.018, -0.018)   # right

        # ── Mouth ───────────────────────────────────────────────
        # Philtrum groove — indent above upper lip
        z += G(0.500, 0.662, 0.022, 0.022, -0.020)
        # Upper lip — cupid's bow
        z += G(0.500, 0.698, 0.100, 0.026, +0.068)
        # Lip groove / separation line
        z += G(0.500, 0.716, 0.088, 0.010, -0.016)
        # Lower lip — fuller
        z += G(0.500, 0.738, 0.092, 0.032, +0.078)
        # Chin dimple
        z += G(0.500, 0.820, 0.028, 0.022, -0.015)

        # ── Cheekbones ──────────────────────────────────────────
        z += G(0.215, 0.500, 0.082, 0.095, +0.042)   # left
        z += G(0.785, 0.500, 0.082, 0.095, +0.042)   # right

        # ── Nasolabial folds — subtle groove cheek-to-mouth ─────
        z += G(0.400, 0.600, 0.022, 0.060, -0.012)   # left
        z += G(0.600, 0.600, 0.022, 0.060, -0.012)   # right

        return z

    # ── parametric surface ────────────────────────────────────────────

    def _parametric(self, u: float, v: float):
        """Map (u,v) -> (x,y,z) on anime face surface."""
        # Phi range 18..180 deg: round crown (not a pole-point), V-chin
        # Theta ±50 deg: front-facing, no edge wrap artifacts
        theta = (u - 0.5) * math.pi * 0.56
        phi   = math.pi * (0.10 + v * 0.90)

        sin_phi   = math.sin(phi)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        w = self._anime_width(v)

        x_base = self.width * w * sin_phi * sin_theta
        z_base = self.depth * sin_phi * cos_theta

        rs = self._eye_socket_scale(u, v)
        x  = x_base * rs
        z  = z_base * rs + self.depth * self._anime_features(u, v)
        y  = self.height * (0.52 - v)

        return float(x), float(y), float(z)

    def _normal(self, u: float, v: float):
        """Outward surface normal via finite difference."""
        eps = 0.001
        x0, y0, z0 = self._parametric(u, v)
        xu, yu, zu = self._parametric(min(u + eps, 1.0), v)
        xv, yv, zv = self._parametric(u, min(v + eps, 1.0))

        du = np.array([xu - x0, yu - y0, zu - z0])
        dv = np.array([xv - x0, yv - y0, zv - z0])

        n  = np.cross(dv, du)   # dv x du → outward-facing
        ln = np.linalg.norm(n)
        if ln < 1e-8:
            return 0.0, 0.0, 1.0
        n /= ln
        return float(n[0]), float(n[1]), float(n[2])

    def _label(self, idx: int, u: float, v: float, groups: dict):
        """Assign vertex to UV-defined face regions."""
        for name, bounds in self.cfg["regions"].items():
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
