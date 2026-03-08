"""
renderer.py — RAEON Face Renderer

Pure ModernGL + PyGLM. No game engine. No Three.js. Ours entirely.
Phong shading with skin-tuned lighting.
Expression displacements uploaded to GPU every frame.
Head rotation via model matrix (tilt + gaze).
"""

import numpy as np
import glm
import moderngl


VERT_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec3 in_displacement;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

out vec3 v_normal;
out vec3 v_frag_pos;

void main() {
    vec3 pos    = in_position + in_displacement;
    vec4 world  = u_model * vec4(pos, 1.0);

    gl_Position = u_proj * u_view * world;
    v_frag_pos  = vec3(world);
    v_normal    = normalize(u_normal_mat * (in_normal + in_displacement * 0.3));
}
"""

FRAG_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_frag_pos;

uniform vec3 u_light_pos;
uniform vec3 u_light_color;
uniform vec3 u_view_pos;
uniform vec3 u_skin_color;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;

out vec4 out_color;

void main() {
    vec3 norm      = normalize(v_normal);
    vec3 light_dir = normalize(u_light_pos - v_frag_pos);
    vec3 view_dir  = normalize(u_view_pos  - v_frag_pos);
    vec3 half_dir  = normalize(light_dir + view_dir);

    // Ambient
    vec3 ambient = u_ambient * u_skin_color;

    // Diffuse
    float diff   = max(dot(norm, light_dir), 0.0);
    vec3 diffuse = u_diffuse * diff * u_light_color * u_skin_color;

    // Specular (Blinn-Phong)
    float spec   = pow(max(dot(norm, half_dir), 0.0), u_shininess);
    vec3 specular = u_specular * spec * u_light_color;

    // Subsurface scatter approximation (cheap)
    float sss    = 0.08 * max(dot(-norm, light_dir), 0.0);
    vec3 scatter = sss * vec3(0.95, 0.55, 0.45) * u_skin_color;

    // Rim lighting — anime-style cool blue edge glow
    float rim      = pow(1.0 - max(dot(norm, view_dir), 0.0), 4.0);
    vec3 rim_light = 0.25 * rim * vec3(0.70, 0.85, 1.0);

    vec3 color   = ambient + diffuse + specular + scatter + rim_light;
    out_color    = vec4(color, 1.0);
}
"""


class FaceRenderer:

    def __init__(self, ctx: moderngl.Context, mesh, face_cfg: dict):
        self.ctx      = ctx
        self.mesh     = mesh
        self.face_cfg = face_cfg

        self._build_program()
        self._build_buffers()
        self._disp_buffer = None
        self._init_disp_buffer()

        # Camera + light
        cam  = face_cfg["camera"]
        self.view_pos = glm.vec3(*cam["position"])
        self.view = glm.lookAt(
            self.view_pos,
            glm.vec3(0, 0, 0),
            glm.vec3(0, 1, 0)
        )
        self._view_t = glm.transpose(self.view)   # pre-transposed for upload

        light = face_cfg["light"]
        self.light_pos   = glm.vec3(*light["position"])
        self.light_color = glm.vec3(*light["color"])

        skin = face_cfg["skin"]
        self.skin_color = glm.vec3(*skin["base_color"])
        self.ambient    = skin["ambient"]
        self.diffuse    = skin["diffuse"]
        self.specular   = skin["specular"]
        self.shininess  = skin["shininess"]

        # Head transform
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
        disps  = np.zeros_like(verts)
        idxs   = self.mesh.indices

        self.vbo_pos  = self.ctx.buffer(verts.astype("f4").tobytes())
        self.vbo_norm = self.ctx.buffer(norms.astype("f4").tobytes())
        self.vbo_disp = self.ctx.buffer(disps.astype("f4").tobytes())
        self.ibo      = self.ctx.buffer(idxs.astype("i4").tobytes())

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.vbo_pos,  "3f", "in_position"),
                (self.vbo_norm, "3f", "in_normal"),
                (self.vbo_disp, "3f", "in_displacement"),
            ],
            self.ibo,
        )

    def _init_disp_buffer(self):
        n = self.mesh.vertex_count()
        self._disp_data = np.zeros((n, 3), dtype=np.float32)

    # ── per-frame ────────────────────────────────────────────────────

    def update_displacements(self, displacements: np.ndarray):
        """Upload new displacement array to GPU. Called every frame."""
        self._disp_data = displacements.astype(np.float32)
        self.vbo_disp.write(self._disp_data.flatten().tobytes())

    def update_head_rotation(self, tilt_deg: float, gaze_deg: float):
        self.tilt_deg = tilt_deg
        self.gaze_deg = gaze_deg

    def render(self, width: int, height: int):
        self.ctx.clear(0.08, 0.08, 0.10, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        # Build model matrix: head tilt + gaze
        model = glm.mat4(1.0)
        model = glm.rotate(model, glm.radians(self.tilt_deg), glm.vec3(0, 0, 1))
        model = glm.rotate(model, glm.radians(self.gaze_deg), glm.vec3(0, 1, 0))

        # Projection
        aspect = width / max(height, 1)
        proj   = glm.perspective(
            glm.radians(self.face_cfg["camera"]["fov"]),
            aspect,
            self.face_cfg["camera"]["near"],
            self.face_cfg["camera"]["far"],
        )

        normal_mat = glm.mat3(glm.transpose(glm.inverse(model)))

        # Set uniforms — PyGLM bytes() is row-major; OpenGL needs column-major.
        # Transpose mat4/mat3 before bytes() to fix the memory layout.
        self.prog["u_model"].write(bytes(glm.transpose(model)))
        self.prog["u_view"].write(bytes(self._view_t))
        self.prog["u_proj"].write(bytes(glm.transpose(proj)))
        self.prog["u_normal_mat"].write(bytes(glm.transpose(normal_mat)))

        # vec3 uniforms are fine — no row/column ambiguity
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
        self.vbo_disp.release()
        self.ibo.release()
        self.vao.release()
        self.prog.release()
