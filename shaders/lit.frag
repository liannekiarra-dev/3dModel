#version 330

uniform vec3 u_light_dir;
uniform vec3 u_camera_pos;
uniform vec3 u_base_color;
uniform float u_specular_exp;
uniform float u_ambient;
uniform float u_kd;
uniform float u_ks;
uniform sampler2D u_shadow_map;
uniform vec2 u_shadow_texel;
uniform int u_use_pcf;

in VS_OUT {
    vec3 world_pos;
    vec3 world_normal;
    vec4 light_clip;
} fs_in;

out vec4 f_color;

float sample_shadow(vec2 uv, float current) {
    float closest = texture(u_shadow_map, uv).r;
    return current > closest ? 0.0 : 1.0;
}

float shadow_visibility(vec3 proj_coords) {
    float ndotl = dot(normalize(fs_in.world_normal), -u_light_dir);
    float bias = max(0.002 * (1.0 - ndotl), 0.0005);
    float current = proj_coords.z - bias;
    vec2 uv = proj_coords.xy;

    if (u_use_pcf != 0) {
        float sum = 0.0;
        vec2 texel = u_shadow_texel;
        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                vec2 suv = uv + vec2(x, y) * texel;
                sum += sample_shadow(suv, current);
            }
        }
        return sum / 9.0;
    }

    return sample_shadow(uv, current);
}

void main() {
    vec3 N = normalize(fs_in.world_normal);
    vec3 L = normalize(-u_light_dir);
    float ndl = max(dot(N, L), 0.0);

    vec3 V = normalize(u_camera_pos - fs_in.world_pos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), u_specular_exp);

    vec4 lcs = fs_in.light_clip;
    vec3 proj = lcs.xyz * 0.5 + 0.5;
    float vis = 1.0;
    if (proj.x >= 0.0 && proj.x <= 1.0 && proj.y >= 0.0 && proj.y <= 1.0 && proj.z <= 1.0) {
        vis = shadow_visibility(proj);
    }

    vec3 diffuse = u_kd * u_base_color * ndl;
    vec3 specular = u_ks * vec3(1.0) * spec;
    vec3 ambient = u_ambient * u_base_color;
    vec3 color = ambient + vis * (diffuse + specular);
    f_color = vec4(color, 1.0);
}
