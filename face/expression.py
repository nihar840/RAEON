"""
expression.py — RAEON Expression Engine

ExpressionVector: 10-dimensional emotion state (brain contract — unchanged).
ExpressionEngine: lerps current → target each frame, returns dict for shader.

SDF renderer reads expression values directly as uniforms — no vertex
displacements, no group caching, no mesh dependency.
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass, field


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
    Smooth expression blending engine.
    Lerps current → target at 8% per frame, returns expression dict
    for the SDF renderer to upload as uniforms.
    """

    INTERP = {
        "linear":  lambda x: x,
        "smooth":  lambda x: x * x * (3 - 2 * x),
        "elastic": lambda x: math.sin(x * math.pi / 2),
    }

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "expressions.json"
        with open(config_path) as f:
            self.cfg = json.load(f)

        self._presets = self.cfg.get("emotion_presets", {})
        self.current  = ExpressionVector()
        self.target   = ExpressionVector()

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

    def get_expression_dict(self) -> dict:
        """Return current expression state as a flat dict for the renderer."""
        return self.current.to_dict()

    def presets(self) -> list:
        return list(self._presets.keys())
