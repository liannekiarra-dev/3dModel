
"""
This is main.py, this contains all functionalities to automate the build of the 3D torus.
The shaders folder contains the vertex and fragment shader coding for the model.

Real-time Blinn-Phong shading with directional shadow mapping (depth pass + lit pass).

Optional 3x3 PCF toggles with the P key.
"""

from __future__ import annotations

from pathlib import Path

import glm
import moderngl as mgl
import moderngl_window as mglw
import numpy as np

from meshes import make_plane, make_torus


def _vec3_bytes(v: glm.vec3) -> bytes:
    return np.array([v.x, v.y, v.z], dtype=np.float32).tobytes()


def _vec2_bytes(v: glm.vec2) -> bytes:
    return np.array([v.x, v.y], dtype=np.float32).tobytes()


class ShadowApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Blinn-Phong + Shadow Mapping for 3D Torus"
    window_size = (1280, 720)
    aspect_ratio = 16 / 9
    resizable = True
    samples = 4

    resource_dir = Path(__file__).resolve().parent

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.CULL_FACE)
        self.ctx.cull_face = "back"

        self.shadow_size = 2048
        self.depth_tex = self.ctx.depth_texture((self.shadow_size, self.shadow_size))
        self.depth_tex.filter = (mgl.NEAREST, mgl.NEAREST)
        self.depth_tex.repeat_x = False
        self.depth_tex.repeat_y = False
        self.shadow_fbo = self.ctx.framebuffer(depth_attachment=self.depth_tex)

        shader_dir = self.resource_dir / "shaders"
        self.prog_depth = self.ctx.program(
            vertex_shader=(shader_dir / "depth.vert").read_text(),
            fragment_shader=(shader_dir / "depth.frag").read_text(),
        )
        self.prog_lit = self.ctx.program(
            vertex_shader=(shader_dir / "lit.vert").read_text(),
            fragment_shader=(shader_dir / "lit.frag").read_text(),
        )

        plane_pos, plane_n = make_plane(26.0, 0.0)
        torus_pos, torus_n = make_torus()
        
        self.torus_model = glm.translate(glm.mat4(1.0), glm.vec3(0.0, 0.4, 0.0))

        def build_vao(pos: np.ndarray, nrm: np.ndarray) -> mgl.VertexArray:
            vbo = self.ctx.buffer(np.hstack([pos, nrm]).astype("f4").tobytes())
            return self.ctx.vertex_array(
                self.prog_lit,
                [(vbo, "3f 3f", "in_position", "in_normal")],
            )

        self.vao_plane_lit = build_vao(plane_pos, plane_n)
        self.vao_torus_lit = build_vao(torus_pos, torus_n)

        plane_vbo_d = self.ctx.buffer(plane_pos.astype("f4").tobytes())
        torus_vbo_d = self.ctx.buffer(torus_pos.astype("f4").tobytes())
        self.vao_plane_depth = self.ctx.vertex_array(
            self.prog_depth,
            [(plane_vbo_d, "3f", "in_position")],
        )
        self.vao_torus_depth = self.ctx.vertex_array(
            self.prog_depth,
            [(torus_vbo_d, "3f", "in_position")],
        )

       
        self.light_ray = glm.normalize(glm.vec3(0.42, -0.88, 0.25))
        self.L_to_light = -self.light_ray
        eye_light = glm.vec3(0.0, 2.0, 0.0) - self.light_ray * 32.0
        center = glm.vec3(0.0, 1.0, 0.0)
        self.light_view = glm.lookAt(eye_light, center, glm.vec3(0.0, 1.0, 0.0))
        self.light_proj = glm.ortho(-13.0, 13.0, -13.0, 13.0, 2.0, 55.0)

        self.cam_yaw = 0.55
        self.cam_pitch = -0.38
        self.cam_dist = 13.5
        self.cam_target = glm.vec3(0.0, 0.55, 0.0)

        self.use_pcf = True
        self.wireframe = False

    def _camera_view(self) -> glm.mat4:
        x = self.cam_dist * np.cos(self.cam_pitch) * np.sin(self.cam_yaw)
        y = self.cam_dist * np.sin(self.cam_pitch)
        z = self.cam_dist * np.cos(self.cam_pitch) * np.cos(self.cam_yaw)
        eye = self.cam_target + glm.vec3(x, y, z)
        return glm.lookAt(eye, self.cam_target, glm.vec3(0.0, 1.0, 0.0))

    def _proj(self) -> glm.mat4:
        return glm.perspective(glm.radians(55.0), self.wnd.aspect_ratio, 0.1, 120.0)

    def on_render(self, time: float, frame_time: float):
        view = self._camera_view()
        proj = self._proj()
        light_vp = self.light_proj * self.light_view

        
        self.shadow_fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
        self.ctx.clear(depth=1.0)
        self.prog_depth["mvp"].write((light_vp * glm.mat4(1.0)).to_bytes())
        self.vao_plane_depth.render()
        self.prog_depth["mvp"].write((light_vp * self.torus_model).to_bytes())
        self.vao_torus_depth.render()

        
        self.wnd.fbo.use()
        self.ctx.viewport = (0, 0, *self.wnd.size)
        self.ctx.clear(0.08, 0.09, 0.12, 1.0)
        self.depth_tex.use(location=0)
        self.prog_lit["u_shadow_map"].value = 0
        self.prog_lit["u_light_dir"].write(_vec3_bytes(self.L_to_light))
        cam_eye = self.cam_target + glm.vec3(
            self.cam_dist * np.cos(self.cam_pitch) * np.sin(self.cam_yaw),
            self.cam_dist * np.sin(self.cam_pitch),
            self.cam_dist * np.cos(self.cam_pitch) * np.cos(self.cam_yaw),
        )
        self.prog_lit["u_camera_pos"].write(_vec3_bytes(cam_eye))
        self.prog_lit["u_specular_exp"].value = 48.0
        self.prog_lit["u_ambient"].value = 0.12
        self.prog_lit["u_use_pcf"].value = 1 if self.use_pcf else 0
        texel = glm.vec2(1.0 / float(self.shadow_size))
        self.prog_lit["u_shadow_texel"].write(_vec2_bytes(texel))

        self.prog_lit["projection"].write(proj.to_bytes())
        self.prog_lit["view"].write(view.to_bytes())

        ident = glm.mat4(1.0)
        ls = light_vp * ident
        self.prog_lit["light_space"].write(ls.to_bytes())
        self.prog_lit["model"].write(ident.to_bytes())
        self.prog_lit["u_base_color"].write(_vec3_bytes(glm.vec3(0.95, 0.95, 0.95))) #color toggle - plane
        self.prog_lit["u_kd"].value = 0.85
        self.prog_lit["u_ks"].value = 0.08
        self.vao_plane_lit.render()

        ls = light_vp * self.torus_model
        self.prog_lit["light_space"].write(ls.to_bytes())
        self.prog_lit["model"].write(self.torus_model.to_bytes())
        self.prog_lit["u_base_color"].write(_vec3_bytes(glm.vec3(0.75, 0.35, 0.18))) #color toggle - torus
        self.prog_lit["u_kd"].value = 0.9
        self.prog_lit["u_ks"].value = 0.55
        self.vao_torus_lit.render()

    def on_key_event(self, key: int, action: int, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.P:
            self.use_pcf = not self.use_pcf
            mode = "PCF 3x3" if self.use_pcf else "hard shadow"# P(key) toggle for PCF or hard shadow
            print(f"Shadow filter: {mode}")
        if key == self.wnd.keys.W:
            self.wireframe = not self.wireframe
            self.ctx.wireframe = self.wireframe
            print(f"Wireframe: {self.wireframe}")
        if key == self.wnd.keys.ESCAPE:
            self.wnd.close()

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        self.cam_yaw += dx * 0.006
        self.cam_pitch = float(np.clip(self.cam_pitch - dy * 0.006, -1.2, 1.2))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.cam_dist = float(np.clip(self.cam_dist - y_offset * 0.7, 4.0, 40.0))


if __name__ == "__main__":
    mglw.run_window_config(ShadowApp)
