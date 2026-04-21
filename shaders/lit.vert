#version 330

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 light_space;

in vec3 in_position;
in vec3 in_normal;

out VS_OUT {
    vec3 world_pos;
    vec3 world_normal;
    vec4 light_clip;
} vs_out;

void main() {
    vec4 wp = model * vec4(in_position, 1.0);
    vs_out.world_pos = wp.xyz;
    mat3 normal_mat = mat3(transpose(inverse(model)));
    vs_out.world_normal = normalize(normal_mat * in_normal);
    vs_out.light_clip = light_space * vec4(in_position, 1.0);
    gl_Position = projection * view * wp;
}
