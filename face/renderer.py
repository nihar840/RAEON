"""
renderer.py — RAEON Face Renderer

Pure ModernGL + PyGLM. No game engine. No Three.js. Ours entirely.

Key design: procedural face colour computed in the FRAGMENT SHADER via
interpolated UV coordinates.  This gives pixel-level precision for eyes,
brows, and lips regardless of mesh vertex resolution.

Colour layers (evaluated per-pixel from UV):
  u_skin_color → lips → sclera → iris → pupil → highlight → eyebrows

Lighting: Phong + subsurface scatter + rim light.
"""

import numpy as np
import glm
import moderngl


VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec3 in_displacement;
in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

out vec3 v_normal;
out vec3 v_frag_pos;
out vec2 v_uv;

void main() {
    vec3 pos    = in_position + in_displacement;
    vec4 world  = u_model * vec4(pos, 1.0);

    gl_Position = u_proj * u_view * world;
    v_frag_pos  = vec3(world);
    v_normal    = normalize(u_normal_mat * (in_normal + in_displacement * 0.3));
    v_uv        = in_uv;
}
"""

FRAG_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_frag_pos;
in vec2 v_uv;

uniform vec3  u_light_pos;
uniform vec3  u_light_color;
uniform vec3  u_view_pos;
uniform vec3  u_skin_color;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;

out vec4 out_color;

// ── Gaussian helper ─────────────────────────────────────────────────
// su, sv are standard deviations in UV space (u-space scaled for circles)
float G(vec2 uv, vec2 c, float su, float sv) {
    vec2 d = (uv - c) / vec2(su, sv);
    return exp(-0.5 * dot(d, d));
}

// ── Procedural face colour from UV ──────────────────────────────────
// All Gaussian sizes use su ≈ sv * 1.55 to produce circles on-screen.
// (At eye level: dx/du ≈ 0.57, dy/dv ≈ 0.88 → ratio = 0.88/0.57 = 1.54)
//
// Layer order: skin → lips → sclera → iris → pupil → highlight → brow
vec3 face_material(vec2 uv) {
    vec3  c = u_skin_color;
    float a;

    // ── Lips ────────────────────────────────────────────────────────
    a = clamp(G(uv, vec2(0.500, 0.695), 0.130, 0.040) * 0.90
            + G(uv, vec2(0.500, 0.735), 0.118, 0.036) * 1.00,
            0.0, 1.0);
    c = mix(c, vec3(0.82, 0.47, 0.45), a);

    // ── Left eye ────────────────────────────────────────────────────
    // Sclera (off-white almond) — wider so it's clearly visible
    a = clamp(G(uv, vec2(0.335, 0.385), 0.062, 0.040) * 2.4, 0.0, 1.0);
    c = mix(c, vec3(0.93, 0.91, 0.89), a);
    // Iris (warm dark hazel)
    a = clamp(G(uv, vec2(0.335, 0.386), 0.029, 0.019) * 2.6, 0.0, 1.0);
    c = mix(c, vec3(0.23, 0.14, 0.09), a);
    // Pupil (near-black)
    a = clamp(G(uv, vec2(0.335, 0.387), 0.016, 0.010) * 3.8, 0.0, 1.0);
    c = mix(c, vec3(0.04, 0.03, 0.02), a);
    // Specular highlight (bright crescent top-left)
    a = clamp(G(uv, vec2(0.326, 0.376), 0.009, 0.006) * 5.5, 0.0, 1.0);
    c = mix(c, vec3(0.96, 0.96, 1.00), a);

    // ── Right eye ───────────────────────────────────────────────────
    // Sclera
    a = clamp(G(uv, vec2(0.665, 0.385), 0.062, 0.040) * 2.4, 0.0, 1.0);
    c = mix(c, vec3(0.93, 0.91, 0.89), a);
    // Iris
    a = clamp(G(uv, vec2(0.665, 0.386), 0.029, 0.019) * 2.6, 0.0, 1.0);
    c = mix(c, vec3(0.23, 0.14, 0.09), a);
    // Pupil
    a = clamp(G(uv, vec2(0.665, 0.387), 0.016, 0.010) * 3.8, 0.0, 1.0);
    c = mix(c, vec3(0.04, 0.03, 0.02), a);
    // Highlight
    a = clamp(G(uv, vec2(0.674, 0.376), 0.009, 0.006) * 5.5, 0.0, 1.0);
    c = mix(c, vec3(0.96, 0.96, 1.00), a);

    // ── Left eyebrow (arch: body + inner head + outer tail) ─────────
    a = clamp(G(uv, vec2(0.326, 0.262), 0.060, 0.016) * 2.8   // main arch
            + G(uv, vec2(0.290, 0.268), 0.022, 0.014) * 2.0   // inner head
            + G(uv, vec2(0.360, 0.258), 0.022, 0.013) * 1.8,  // outer tail
            0.0, 1.0);
    c = mix(c, vec3(0.10, 0.07, 0.04), a);

    // ── Right eyebrow (mirrored) ────────────────────────────────────
    a = clamp(G(uv, vec2(0.674, 0.262), 0.060, 0.016) * 2.8
            + G(uv, vec2(0.710, 0.268), 0.022, 0.014) * 2.0
            + G(uv, vec2(0.640, 0.258), 0.022, 0.013) * 1.8,
            0.0, 1.0);
    c = mix(c, vec3(0.10, 0.07, 0.04), a);

    return c;
}

void main() {
    vec3 norm      = normalize(v_normal);
    vec3 light_dir = normalize(u_light_pos - v_frag_pos);
    vec3 view_dir  = normalize(u_view_pos  - v_frag_pos);
    vec3 half_dir  = normalize(light_dir + view_dir);

    // Material colour from procedural UV map
    vec3 mat = face_material(v_uv);

    // How "skin-like" is this pixel? (1=skin, 0=eye/brow/lip)
    float is_skin = clamp(dot(mat - u_skin_color, mat - u_skin_color) < 0.005
                          ? 1.0 : 0.0, 0.0, 1.0);
    // Approximate: pixels close to skin_color are skin
    float skin_w = 1.0 - clamp(length(mat - u_skin_color) * 4.0, 0.0, 1.0);

    // Ambient
    vec3 ambient = u_ambient * mat;

    // Diffuse — eye-socket fill prevents concave shadow from hiding sclera
    float diff   = max(dot(norm, light_dir), 0.0);
    float e_fill = clamp(G(v_uv, vec2(0.335, 0.385), 0.095, 0.062)
                       + G(v_uv, vec2(0.665, 0.385), 0.095, 0.062), 0.0, 0.90);
    diff = max(diff, 0.42 * e_fill);
    vec3 diffuse = u_diffuse * diff * u_light_color * mat;

    // Specular — reduced for non-skin (eyes/lips less shiny)
    float spec_str = mix(0.10, u_specular, skin_w);
    float spec     = pow(max(dot(norm, half_dir), 0.0), u_shininess);
    vec3  specular = spec_str * spec * u_light_color;

    // Subsurface scatter — skin only
    float sss    = 0.08 * skin_w * max(dot(-norm, light_dir), 0.0);
    vec3 scatter = sss * vec3(0.95, 0.55, 0.45) * mat;

    // Rim lighting — anime-style cool blue edge glow
    float rim      = pow(1.0 - max(dot(norm, view_dir), 0.0), 4.0);
    vec3 rim_light = 0.20 * rim * vec3(0.68, 0.84, 1.0);

    vec3 color = ambient + diffuse + specular + scatter + rim_light;
    out_color  = vec4(color, 1.0);
}
"""


class FaceRenderer:

    def __init__(self, ctx: moderngl.Context, mesh, face_cfg: dict):
        self.ctx      = ctx
        self.mesh     = mesh
        self.face_cfg = face_cfg

        self._build_program()
        self._build_buffers()
        self._init_disp_buffer()

        # Camera + light
        cam  = face_cfg["camera"]
        self.view_pos = glm.vec3(*cam["position"])
        self.view = glm.lookAt(
            self.view_pos,
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0)
        )
        self._view_t = glm.transpose(self.view)

        light = face_cfg["light"]
        self.light_pos   = glm.vec3(*light["position"])
        self.light_color = glm.vec3(*light["color"])

        skin = face_cfg["skin"]
        self.skin_color = glm.vec3(*skin["base_color"])
        self.ambient    = skin["ambient"]
        self.diffuse    = skin["diffuse"]
        self.specular   = skin["specular"]
        self.shininess  = skin["shininess"]

        self.tilt_deg = 0.0
        self.gaze_deg = 0.0

    # ── setup ────────────────────────────────────────────────────────

    def _build_program(self):
        self.prog = self.ctx.program(
            vertex_shader=VERT_SHADER,
            fragment_shader=FRAG_SHADER,
        )

    def _build_buffers(self):
        verts  = self.mesh.vertices.flatten()
        norms  = self.mesh.normals.flatten()
        uvs    = self.mesh.uvs.flatten()
        disps  = np.zeros_like(verts)
        idxs   = self.mesh.indices

        self.vbo_pos  = self.ctx.buffer(verts.astype("f4").tobytes())
        self.vbo_norm = self.ctx.buffer(norms.astype("f4").tobytes())
        self.vbo_uv   = self.ctx.buffer(uvs.astype("f4").tobytes())
        self.vbo_disp = self.ctx.buffer(disps.astype("f4").tobytes())
        self.ibo      = self.ctx.buffer(idxs.astype("i4").tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_pos,  "3f", "in_position"),
                (self.vbo_norm, "3f", "in_normal"),
                (self.vbo_disp, "3f", "in_displacement"),
                (self.vbo_uv,   "2f", "in_uv"),
            ],
            self.ibo,
        )

    def _init_disp_buffer(self):
        n = self.mesh.vertex_count()
        self._disp_data = np.zeros((n, 3), dtype=np.float32)

    # ── per-frame ────────────────────────────────────────────────────

    def update_displacements(self, displacements: np.ndarray):
        self._disp_data = displacements.astype(np.float32)
        self.vbo_disp.write(self._disp_data.flatten().tobytes())

    def update_head_rotation(self, tilt_deg: float, gaze_deg: float):
        self.tilt_deg = tilt_deg
        self.gaze_deg = gaze_deg

    def render(self, width: int, height: int):
        self.ctx.clear(0.07, 0.07, 0.09, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(self.tilt_deg), glm.vec3(0, 0, 1))
        model = glm.rotate(model, glm.radians(self.gaze_deg), glm.vec3(0, 1, 0))

        aspect = width / max(height, 1)
        proj   = glm.perspective(
            glm.radians(self.face_cfg["camera"]["fov"]),
            aspect,
            self.face_cfg["camera"]["near"],
            self.face_cfg["camera"]["far"],
        )

        normal_mat = glm.mat3(glm.transpose(glm.inverse(model)))

        self.prog["u_model"].write(bytes(glm.transpose(model)))
        self.prog["u_view"].write(bytes(self._view_t))
        self.prog["u_proj"].write(bytes(glm.transpose(proj)))
        self.prog["u_normal_mat"].write(bytes(glm.transpose(normal_mat)))

        self.prog["u_light_pos"].write(bytes(self.light_pos))
        self.prog["u_light_color"].write(bytes(self.light_color))
        self.prog["u_view_pos"].write(bytes(self.view_pos))
        self.prog["u_skin_color"].write(bytes(self.skin_color))

        self.prog["u_ambient"]   = self.ambient
        self.prog["u_diffuse"]   = self.diffuse
        self.prog["u_specular"]  = self.specular
        self.prog["u_shininess"] = self.shininess

        self.vao.render(moderngl.TRIANGLES)

    def destroy(self):
        self.vbo_pos.release()
        self.vbo_norm.release()
        self.vbo_uv.release()
        self.vbo_disp.release()
        self.ibo.release()
        self.vao.release()
        self.prog.release()
