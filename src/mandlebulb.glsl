#version 460

#include <common.glsl>

struct Uniforms {
    mat4 world_to_screen;
    vec4 eye;
    vec4 fwd;
    vec4 right;
    vec4 up;
    Mouse mouse;
    vec4 background_color;
    uint frame;
    float time;
    float deltatime;
    uint width;
    uint height;
    uint monitor_width;
    uint monitor_height;
    uint march_iterations;
    float t_max;
    float dt_min;
};

layout(set = 0, binding = 0) uniform Ubo {
    Uniforms ubo;
};

float map(vec3 pos) {
    return length(pos - 0.0) - 2.0;
}

vec4 march(vec3 ro, vec3 rd) {
    float t = 0.0;
    for (int i = 0; i<ubo.march_iterations && t<ubo.t_max; i++) {
        vec3 pos = ro + t*rd;
        float dt = map(pos);
        if (dt < ubo.dt_min) {
            break;
        }
        t += dt;
    }
    return vec4(1.0/t, 0.0, 0.0, 1.0);
}

#ifdef EYEFACE_COMPUTE
    layout(set = 0, binding = 1, rgba16) writeonly uniform image2D screen;
#endif // EYEFACE_COMPUTE

#ifdef EYEFACE_DRAW
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y) * int(gl_WorkGroupSize.x) + int(gl_WorkGroupID.x) * 64;

        if (ubo.width * ubo.height <= id) {
            return;
        }
        ivec2 ipos = to2D(id, int(ubo.width));
        ivec2 ires = ivec2(ubo.width, ubo.height);
        vec2 pos = vec2(ipos);
        vec2 res = vec2(ires);

        vec3 ro = ubo.eye.xyz;
        vec3 rd = ubo.fwd.xyz;
        rd += ubo.right.xyz * (pos.x - res.x/2.0)/res.y;
        rd += ubo.up.xyz * (pos.y - res.y/2.0)/res.y;
        rd = normalize(rd);

        vec4 f_color = march(ro, rd);
        imageStore(screen, ipos, f_color);
    }
#endif // EYEFACE_DRAW
