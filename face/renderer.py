"""
renderer.py — RAEON Face Renderer (SDF Emoji Avatar)

Draws the entire face on a fullscreen quad using Signed Distance Functions.
Expression vector (10 floats) passed as shader uniforms — each dimension
continuously controls a visual parameter (eyelid height, mouth curve, etc.).

No mesh, no vertex displacements, no textures, no external assets.
Pixel-perfect, resolution-independent, extremely lightweight.
"""

import numpy as np
import moderngl

# ─────────────────────────────────────────────────────────────────────
# Shaders
# ─────────────────────────────────────────────────────────────────────

VERT_SHADER = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_uv = in_position * 0.5 + 0.5;
}
"""

FRAG_SHADER = """
#version 330 core
in  vec2 v_uv;
out vec4 out_color;

// ── Expression uniforms (10 floats from ExpressionEngine) ──────────
uniform float u_eye_openness;    // 0..1
uniform float u_eyebrow_angle;   // -1..1
uniform float u_brow_scrunch;    // 0..1
uniform float u_lip_curve;       // -1..1  (+ smile, - frown)
uniform float u_lip_part;        // 0..1
uniform float u_jaw_tension;     // -1..1  (+ clench, - drop)
uniform float u_nose_flare;      // 0..1
uniform float u_cheek_raise;     // 0..1
uniform float u_head_tilt;       // -1..1
uniform float u_gaze_direction;  // -1..1

// ── Colour uniforms ────────────────────────────────────────────────
uniform vec3  u_skin_color;
uniform vec3  u_blush_color;
uniform vec3  u_lip_color;
uniform vec3  u_brow_color;
uniform vec3  u_sclera_color;
uniform vec3  u_iris_color;
uniform vec3  u_pupil_color;
uniform vec3  u_highlight_color;
uniform vec3  u_teeth_color;
uniform vec3  u_bg_color;
uniform float u_aspect;          // width / height

// ── SDF primitives ─────────────────────────────────────────────────

float sdCircle(vec2 p, vec2 c, float r) {
    return length(p - c) - r;
}

float sdEllipse(vec2 p, vec2 c, vec2 r) {
    vec2 d = (p - c) / r;
    return (length(d) - 1.0) * min(r.x, r.y);
}

float sdSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdRoundedRect(vec2 p, vec2 c, vec2 hs, float r) {
    vec2 d = abs(p - c) - hs + r;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - r;
}

// Anti-aliased SDF fill: 1 inside, 0 outside, smooth edge
float fill(float d, float edge) {
    return 1.0 - smoothstep(-edge, edge, d);
}

// ── Main ───────────────────────────────────────────────────────────

void main() {
    // Aspect-correct coords: center at (0,0), y range ~ -0.5..0.5
    vec2 uv = v_uv - 0.5;
    uv.x *= u_aspect;

    // HEAD TILT — rotate entire coordinate space
    float tilt_a = u_head_tilt * 0.26;   // ~15 deg max
    float ca = cos(tilt_a), sa = sin(tilt_a);
    uv = mat2(ca, -sa, sa, ca) * uv;

    vec3 color = u_bg_color;
    float aa = 0.003;   // anti-alias edge width

    // ════════════════════════════════════════════════════════════════
    // 1. FACE OVAL — egg shape with V-chin taper
    // ════════════════════════════════════════════════════════════════
    vec2 face_c = vec2(0.0, 0.015);
    vec2 face_r = vec2(0.185, 0.25);

    // Taper chin: narrow x-radius as y goes positive (below center)
    float chin_taper = 1.0 - 0.38 * smoothstep(0.0, face_r.y, uv.y - face_c.y);
    // Jaw tension: clench compresses, drop elongates
    float jaw_sq = max(0.0, u_jaw_tension) * 0.010;
    float jaw_drop = max(0.0, -u_jaw_tension) * 0.012;

    float face_d = sdEllipse(uv, face_c,
        vec2(face_r.x * chin_taper - jaw_sq, face_r.y + jaw_drop));
    float face_m = fill(face_d, aa);
    color = mix(color, u_skin_color, face_m);

    // Subtle 3D lighting gradient (lighter top-left, darker bottom-right)
    color += face_m * 0.035 * (-uv.y * 0.8 + uv.x * 0.3);

    // ════════════════════════════════════════════════════════════════
    // 2. NOSE — small line + nostril dots
    // ════════════════════════════════════════════════════════════════
    vec2 nose_tip = vec2(0.0, 0.055);
    float nose_d = sdCircle(uv, nose_tip, 0.009);
    float nose_a = fill(nose_d, aa * 1.5) * 0.25 * face_m;
    color = mix(color, u_skin_color * 0.72, nose_a);

    // Nostrils — spread by nose_flare
    float n_spread = 0.014 + u_nose_flare * 0.008;
    float nl = sdCircle(uv, nose_tip + vec2(-n_spread, 0.006), 0.005);
    float nr = sdCircle(uv, nose_tip + vec2( n_spread, 0.006), 0.005);
    float nostril_a = fill(min(nl, nr), aa) * 0.30 * face_m;
    color = mix(color, u_skin_color * 0.58, nostril_a);

    // ════════════════════════════════════════════════════════════════
    // 3. BLUSH — cheek_raise drives pink circles
    // ════════════════════════════════════════════════════════════════
    float bl_d = sdCircle(uv, vec2(-0.105, 0.06), 0.042);
    float br_d = sdCircle(uv, vec2( 0.105, 0.06), 0.042);
    float blush_a = u_cheek_raise * 0.38 * fill(min(bl_d, br_d), 0.025) * face_m;
    color = mix(color, u_blush_color, blush_a);

    // ════════════════════════════════════════════════════════════════
    // 4. MOUTH — parabolic lip arcs + teeth
    // ════════════════════════════════════════════════════════════════
    vec2 mouth_c = vec2(0.0, 0.115);
    float mw = 0.068;   // half-width of mouth
    float total_part = u_lip_part * 0.028 + max(0.0, -u_jaw_tension) * 0.014;

    // Upper and lower lip Y positions
    float upper_y = mouth_c.y - total_part * 0.5;
    float lower_y = mouth_c.y + total_part * 0.5;
    float curve_k = u_lip_curve * 0.028;

    // Draw lips as parabolic arcs
    float x_norm = clamp(uv.x / mw, -1.0, 1.0);
    float in_mouth = 1.0 - smoothstep(mw * 0.92, mw, abs(uv.x));

    // Upper lip arc
    float upper_target = upper_y - curve_k * x_norm * x_norm;
    float ud = abs(uv.y - upper_target) - 0.0055;
    float upper_a = fill(ud, aa) * in_mouth * face_m;
    color = mix(color, u_lip_color, upper_a);

    // Lower lip arc (slightly thicker)
    float lower_target = lower_y - curve_k * x_norm * x_norm * 0.55;
    float ld = abs(uv.y - lower_target) - 0.0072;
    float lower_a = fill(ld, aa) * in_mouth * face_m;
    color = mix(color, u_lip_color, lower_a);

    // Teeth (visible when mouth open)
    if (total_part > 0.006) {
        float teeth_d = sdRoundedRect(uv, mouth_c,
            vec2(mw * 0.55, total_part * 0.30), 0.004);
        float teeth_a = fill(teeth_d, aa) * face_m
                       * smoothstep(0.005, 0.016, total_part);
        color = mix(color, u_teeth_color, teeth_a);
    }

    // ════════════════════════════════════════════════════════════════
    // 5. EYES (left and right)
    // ════════════════════════════════════════════════════════════════
    for (int side = -1; side <= 1; side += 2) {
        float sx = float(side);
        vec2 eye_c = vec2(sx * 0.068, -0.042);
        vec2 eye_r = vec2(0.034, 0.022);

        // Eyelid clipping — eye_openness controls visible band
        float eo = u_eye_openness;
        float lid_top = eye_c.y - eye_r.y * (eo * 1.5 - 0.7);
        float lid_bot = eye_c.y + eye_r.y * (eo * 1.2 - 0.15);
        // Cheek raise pushes lower lid up
        lid_bot -= u_cheek_raise * 0.007;

        float lid_vis = step(lid_top, uv.y) * step(uv.y, lid_bot);

        // Sclera
        float sclera_d = sdEllipse(uv, eye_c, eye_r);
        float sclera_a = fill(sclera_d, aa) * lid_vis * face_m;
        color = mix(color, u_sclera_color, sclera_a);

        // Iris (shifted by gaze_direction)
        float gaze_x = u_gaze_direction * 0.013;
        vec2 iris_c = eye_c + vec2(gaze_x, 0.001);
        float iris_d = sdCircle(uv, iris_c, 0.012);
        float iris_a = fill(iris_d, aa) * sclera_a;
        color = mix(color, u_iris_color, iris_a);

        // Pupil
        float pupil_d = sdCircle(uv, iris_c, 0.005);
        float pupil_a = fill(pupil_d, aa) * iris_a;
        color = mix(color, u_pupil_color, pupil_a);

        // Specular highlight (opposite gaze side + above)
        vec2 hi_c = iris_c + vec2(-gaze_x * 0.35 - sx * 0.004, -0.005);
        float hi_d = sdCircle(uv, hi_c, 0.0035);
        float hi_a = fill(hi_d, aa) * sclera_a;
        color = mix(color, u_highlight_color, hi_a);

        // Upper eyelid line
        float lid_line_d = abs(uv.y - lid_top);
        float lid_in_x = fill(sdEllipse(uv, eye_c, eye_r * 1.15), aa);
        float lid_a = fill(lid_line_d - 0.001, aa * 0.8) * lid_in_x * face_m * 0.45;
        color = mix(color, u_skin_color * 0.55, lid_a);

        // Lower eyelid subtle line
        float lid_bot_d = abs(uv.y - lid_bot);
        float lid_bot_a = fill(lid_bot_d - 0.0008, aa * 0.8) * lid_in_x * face_m * 0.20;
        color = mix(color, u_skin_color * 0.65, lid_bot_a);
    }

    // ════════════════════════════════════════════════════════════════
    // 6. EYEBROWS (left and right)
    // ════════════════════════════════════════════════════════════════
    for (int side = -1; side <= 1; side += 2) {
        float sx = float(side);
        float by_base = -0.090;
        float by_off  = u_eyebrow_angle * 0.028;
        float scrunch = u_brow_scrunch * 0.016 * sx;

        // Brow as tapered line segment
        vec2 inner = vec2(sx * 0.028 - scrunch,
                          by_base + by_off + u_eyebrow_angle * 0.012);
        vec2 outer = vec2(sx * 0.098,
                          by_base + by_off - u_eyebrow_angle * 0.006);

        // Taper: thicker at inner, thinner at outer
        float along = dot(uv - inner, outer - inner)
                     / dot(outer - inner, outer - inner);
        along = clamp(along, 0.0, 1.0);
        float thickness = mix(0.0062, 0.0028, along);

        float brow_d = sdSegment(uv, inner, outer) - thickness;
        float brow_a = fill(brow_d, aa) * face_m;
        color = mix(color, u_brow_color, brow_a);
    }

    // ════════════════════════════════════════════════════════════════
    // 7. FACE EDGE — soft shadow at face boundary for depth
    // ════════════════════════════════════════════════════════════════
    float edge_shadow = smoothstep(-0.020, 0.0, face_d) * 0.12 * face_m;
    color -= vec3(edge_shadow);

    out_color = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""


# ─────────────────────────────────────────────────────────────────────
# Renderer class
# ─────────────────────────────────────────────────────────────────────

class FaceRenderer:
    """SDF-based face renderer on a fullscreen quad."""

    def __init__(self, ctx: moderngl.Context, face_cfg: dict):
        self.ctx = ctx
        self.face_cfg = face_cfg
        self._build_program()
        self._build_quad()
        self._load_colors()

    # ── build ───────────────────────────────────────────────────────

    def _build_program(self):
        self.prog = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

    def _build_quad(self):
        """Fullscreen quad: 4 corners in NDC, 2 triangles."""
        verts = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 2, 1, 3], dtype=np.int32)

        self.vbo = self.ctx.buffer(verts.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [(self.vbo, "2f", "in_position")],
            self.ibo,
        )

    def _load_colors(self):
        """Read face colours from config and upload as uniforms."""
        skin = self.face_cfg["skin"]
        self._color_map = {
            "u_skin_color":      skin["base_color"],
            "u_blush_color":     skin["blush_color"],
            "u_lip_color":       skin["lip_color"],
            "u_brow_color":      skin["brow_color"],
            "u_sclera_color":    skin["sclera_color"],
            "u_iris_color":      skin["iris_color"],
            "u_pupil_color":     skin["pupil_color"],
            "u_highlight_color": skin["highlight_color"],
            "u_teeth_color":     skin["teeth_color"],
        }
        self._bg_color = self.face_cfg["background"]

    # ── per-frame API ───────────────────────────────────────────────

    def update_expression(self, expr_dict: dict):
        """Upload expression vector as 10 float uniforms."""
        mapping = {
            "eye_openness":   "u_eye_openness",
            "eyebrow_angle":  "u_eyebrow_angle",
            "brow_scrunch":   "u_brow_scrunch",
            "lip_curve":      "u_lip_curve",
            "lip_part":       "u_lip_part",
            "jaw_tension":    "u_jaw_tension",
            "nose_flare":     "u_nose_flare",
            "cheek_raise":    "u_cheek_raise",
            "head_tilt":      "u_head_tilt",
            "gaze_direction": "u_gaze_direction",
        }
        for key, uniform in mapping.items():
            val = float(expr_dict.get(key, 0.0))
            try:
                self.prog[uniform].value = val
            except KeyError:
                pass

    def render(self, width: int, height: int):
        """Clear and draw the SDF face."""
        bg = self._bg_color
        self.ctx.clear(bg[0], bg[1], bg[2], 1.0)
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Upload colours
        for name, rgb in self._color_map.items():
            try:
                self.prog[name].value = tuple(rgb)
            except KeyError:
                pass
        try:
            self.prog["u_bg_color"].value = tuple(bg)
        except KeyError:
            pass

        # Aspect ratio
        self.prog["u_aspect"].value = width / max(height, 1)

        self.vao.render(moderngl.TRIANGLES)

    # ── cleanup ─────────────────────────────────────────────────────

    def destroy(self):
        self.vbo.release()
        self.ibo.release()
        self.vao.release()
        self.prog.release()
