"""
window.py — RAEON Face Window

GLFW window + main render loop.
Keyboard shortcuts to test expressions live.
Accepts expression updates from external thread (CMA / SpaceDB bridge).

Keyboard shortcuts:
  1-7       : Emotion presets (curiosity, joy, focus, surprise, empathy, thinking, neutral)
  R         : Reset to neutral
  ESC / Q   : Quit
  Arrow keys : Manual head tilt / gaze
"""

import time
import threading
import json
import queue
from pathlib import Path

import glfw
import moderngl
import numpy as np

from .mesh       import FaceMesh
from .expression import ExpressionEngine, ExpressionVector
from .renderer   import FaceRenderer


class RaeonWindow:

    PRESET_KEYS = {
        glfw.KEY_1: "curiosity",
        glfw.KEY_2: "joy",
        glfw.KEY_3: "focus",
        glfw.KEY_4: "surprise",
        glfw.KEY_5: "empathy",
        glfw.KEY_6: "thinking",
        glfw.KEY_7: "neutral",
    }

    def __init__(self, width: int = 720, height: int = 900,
                 title: str = "RAEON"):
        self.width   = width
        self.height  = height
        self.title   = title

        self._expr_queue: queue.Queue = queue.Queue()
        self._running = False

        # Load configs
        cfg_dir  = Path(__file__).parent / "config"
        with open(cfg_dir / "face.json") as f:
            self.face_cfg = json.load(f)

    # ── public API (call from any thread) ────────────────────────────

    def push_expression(self, vector: ExpressionVector):
        """Thread-safe: push new expression from CMA / SpaceDB bridge."""
        self._expr_queue.put(vector)

    def push_preset(self, preset_name: str):
        """Thread-safe: push emotion preset by name."""
        self._expr_queue.put(("preset", preset_name))

    def stop(self):
        self._running = False

    # ── main loop ────────────────────────────────────────────────────

    def run(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        win = glfw.create_window(self.width, self.height, self.title, None, None)
        if not win:
            glfw.terminate()
            raise RuntimeError("Window creation failed")

        glfw.make_context_current(win)
        glfw.swap_interval(1)   # vsync

        ctx = moderngl.create_context()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)

        # Build mesh + expression + renderer
        mesh     = FaceMesh()
        engine   = ExpressionEngine(mesh)
        renderer = FaceRenderer(ctx, mesh, self.face_cfg)

        glfw.set_key_callback(win, self._make_key_cb(engine))

        self._running = True
        last_t = time.time()

        print(f"[RAEON] Window open — {self.width}x{self.height}")
        print(f"[RAEON] Presets: 1=curiosity 2=joy 3=focus 4=surprise "
              f"5=empathy 6=thinking 7=neutral  R=reset  ESC=quit")

        while self._running and not glfw.window_should_close(win):
            now  = time.time()
            dt   = now - last_t
            last_t = now

            # Drain expression queue
            while not self._expr_queue.empty():
                item = self._expr_queue.get_nowait()
                if isinstance(item, ExpressionVector):
                    engine.lerp_to(item)
                elif isinstance(item, tuple) and item[0] == "preset":
                    engine.set_preset(item[1])

            # Tick interpolation
            engine.tick(dt)

            # Compute displacements + head rotation
            disps              = engine.compute_displacements()
            tilt, gaze         = engine.get_head_rotation()

            renderer.update_displacements(disps)
            renderer.update_head_rotation(tilt, gaze)

            # Get framebuffer size (handles HiDPI)
            fb_w, fb_h = glfw.get_framebuffer_size(win)
            ctx.viewport = (0, 0, fb_w, fb_h)

            renderer.render(fb_w, fb_h)

            glfw.swap_buffers(win)
            glfw.poll_events()

        renderer.destroy()
        glfw.terminate()
        print("[RAEON] Window closed.")

    def run_threaded(self) -> threading.Thread:
        """Start window in background thread. Returns the thread."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    # ── keyboard ─────────────────────────────────────────────────────

    def _make_key_cb(self, engine: ExpressionEngine):
        def _cb(win, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return

            if key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                self._running = False

            elif key == glfw.KEY_R:
                engine.set_preset("neutral")
                print("[RAEON] Reset to neutral")

            elif key in self.PRESET_KEYS:
                name = self.PRESET_KEYS[key]
                engine.set_preset(name)
                print(f"[RAEON] Preset: {name}")

            elif key == glfw.KEY_LEFT:
                ev = engine.current.to_dict()
                ev["gaze_direction"] = max(-1.0, ev.get("gaze_direction", 0) - 0.1)
                engine.lerp_to(ExpressionVector.from_dict(ev))

            elif key == glfw.KEY_RIGHT:
                ev = engine.current.to_dict()
                ev["gaze_direction"] = min(1.0, ev.get("gaze_direction", 0) + 0.1)
                engine.lerp_to(ExpressionVector.from_dict(ev))

            elif key == glfw.KEY_UP:
                ev = engine.current.to_dict()
                ev["head_tilt"] = max(-1.0, ev.get("head_tilt", 0) - 0.1)
                engine.lerp_to(ExpressionVector.from_dict(ev))

            elif key == glfw.KEY_DOWN:
                ev = engine.current.to_dict()
                ev["head_tilt"] = min(1.0, ev.get("head_tilt", 0) + 0.1)
                engine.lerp_to(ExpressionVector.from_dict(ev))

        return _cb
