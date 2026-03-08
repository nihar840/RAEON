"""
expression.py — RAEON Expression Engine

Maps a CMA expression vector onto vertex displacements.
Fully driven by expressions.json — change the config, change the face.
No hardcoded expressions anywhere.

ExpressionVector (from CMA or emotion preset)
    -> ExpressionEngine.compute(vector)
    -> np.ndarray of shape (N, 3)  -- per-vertex displacement
    -> Sent to GPU each frame
"""

import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExpressionVector:
    """Current state of all RAEON expressions. Fully configurable."""
    eye_openness:   float = 0.6
    eyebrow_angle:  float = 0.0
    brow_scrunch:   float = 0.0
    lip_curve:      float = 0.0
    lip_part:       float = 0.0
    jaw_tension:    float = 0.0
    nose_flare:     float = 0.0
    cheek_raise:    float = 0.0
    head_tilt:      float = 0.0
    gaze_direction: float = 0.0

    # Extra fields for any new expressions added to config dynamically
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items() if k != "extras"}
        d.update(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExpressionVector":
        known = {k for k in cls.__dataclass_fields__ if k != "extras"}
        base   = {k: v for k, v in d.items() if k in known}
        extras = {k: v for k, v in d.items() if k not in known and k != "extras"}
        return cls(**base, extras=extras)

    def blend(self, target: "ExpressionVector", t: float) -> "ExpressionVector":
        """Linearly interpolate toward target. t=0 stay, t=1 snap."""
        t = max(0.0, min(1.0, t))
        td = target.to_dict()
        sd = self.to_dict()
        blended = {k: sd.get(k, 0.0) + t * (td.get(k, 0.0) - sd.get(k, 0.0))
                   for k in set(sd) | set(td)}
        return ExpressionVector.from_dict(blended)


class ExpressionEngine:
    """
    Computes per-vertex displacement arrays from an ExpressionVector.
    Reads config from expressions.json — fully data-driven.
    """

    INTERP = {
        "linear":  lambda x: x,
        "smooth":  lambda x: x * x * (3 - 2 * x),
        "elastic": lambda x: math.sin(x * math.pi / 2),
    }

    def __init__(self, mesh, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "expressions.json"
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.mesh      = mesh
        self._n        = mesh.vertex_count()
        self._exprs    = self.cfg["expressions"]
        self._presets  = self.cfg.get("emotion_presets", {})

        # Current + target state
        self.current   = ExpressionVector()
        self.target    = ExpressionVector()

        # Cache group → vertex index arrays for speed
        self._group_cache = {
            name: np.array(mesh.group_indices(name), dtype=np.int32)
            for name in mesh.groups
        }

        # Head transform state (rotation only, not vertex displacement)
        self.head_tilt_deg  = 0.0
        self.gaze_dir_deg   = 0.0

    # ── public API ───────────────────────────────────────────────────

    def set(self, vector: ExpressionVector):
        """Snap immediately to expression."""
        self.current = self.target = vector

    def set_from_dict(self, d: dict):
        self.set(ExpressionVector.from_dict(d))

    def set_preset(self, name: str):
        """Apply named emotion preset from config."""
        preset = self._presets.get(name)
        if preset:
            self.target = ExpressionVector.from_dict(preset)

    def lerp_to(self, vector: ExpressionVector, speed: float = 0.08):
        """Smooth transition toward target. Call tick() every frame."""
        self.target = vector

    def tick(self, dt: float = 0.016, speed: float = 0.08):
        """Advance interpolation. Call once per frame."""
        self.current = self.current.blend(self.target, speed)
        self._update_head_transforms()

    def compute_displacements(self) -> np.ndarray:
        """
        Returns (N, 3) displacement array for current expression state.
        Add this to base vertex positions before upload to GPU.
        """
        displacements = np.zeros((self._n, 3), dtype=np.float32)
        ev = self.current.to_dict()

        for expr_name, expr_cfg in self._exprs.items():
            if expr_cfg.get("is_head_transform"):
                continue   # handled separately via head matrix

            value = ev.get(expr_name, expr_cfg.get("default", 0.0))
            if abs(value) < 1e-6:
                continue

            # Normalize value to [0..1] or [-1..1] based on range
            lo, hi = expr_cfg["range"]
            norm_val = (value - lo) / (hi - lo + 1e-8)

            # Apply interpolation
            interp_fn = self.INTERP.get(
                expr_cfg.get("interpolation", "smooth"),
                self.INTERP["smooth"]
            )
            norm_val = interp_fn(max(0.0, min(1.0, norm_val)))

            # Center value: 0..1 range maps to -1..+1 effect
            effect = (norm_val * 2.0 - 1.0) if lo < 0 else norm_val

            # Apply per-group displacements
            for group_name, disp_cfg in expr_cfg.get("displacements", {}).items():
                indices = self._group_cache.get(group_name)
                if indices is None or len(indices) == 0:
                    continue

                axis      = disp_cfg["axis"]
                magnitude = disp_cfg["magnitude"]
                mode      = disp_cfg.get("mode", "direct")

                delta = self._compute_delta(
                    indices, axis, magnitude, effect, mode
                )
                displacements[indices] += delta

        return displacements

    def get_head_rotation(self) -> tuple:
        """Returns (tilt_deg, gaze_deg) for head transform matrix."""
        return self.head_tilt_deg, self.gaze_dir_deg

    # ── internal ────────────────────────────────────────────────────

    def _compute_delta(self, indices, axis, magnitude, effect, mode):
        n = len(indices)

        if isinstance(axis, list):
            # Multi-axis displacement
            delta = np.zeros((n, 3), dtype=np.float32)
            for ax in axis:
                i = {"x": 0, "y": 1, "z": 2}[ax]
                delta[:, i] += magnitude * effect
            return delta

        i = {"x": 0, "y": 1, "z": 2}[axis]
        delta = np.zeros((n, 3), dtype=np.float32)

        if mode == "expand":
            # Each vertex moves away from group centroid
            group_verts = self.mesh.vertices[indices]
            centroid    = group_verts.mean(axis=0)
            diff        = group_verts - centroid
            diff_axis   = diff[:, i]
            delta[:, i] = np.sign(diff_axis) * magnitude * effect
        elif mode == "inward":
            delta[:, i] = magnitude * effect
        else:
            delta[:, i] = magnitude * effect

        return delta

    def _update_head_transforms(self):
        ev = self.current.to_dict()
        for expr_name, expr_cfg in self._exprs.items():
            if not expr_cfg.get("is_head_transform"):
                continue
            value    = ev.get(expr_name, 0.0)
            max_deg  = expr_cfg.get("max_degrees", 20.0)
            rot_axis = expr_cfg.get("rotation_axis", "y")
            angle    = value * max_deg
            if rot_axis == "z":
                self.head_tilt_deg = angle
            elif rot_axis == "y":
                self.gaze_dir_deg  = angle

    def presets(self) -> list:
        return list(self._presets.keys())
