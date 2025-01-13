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
    uint voxel_grid_side;
};

layout(set = 0, binding = 0) uniform Ubo {
    Uniforms ubo;
};

float bulb(vec3 pos) {
    const int iterations = 4;
    const float r_cap = 1.2;
    
    vec3 w = pos;
    float dr = 1.0;
    float r2 = dot(w, w);
    
    for (int i = 0; i < iterations && r2 < r_cap * r_cap; i++) {
        // -[](https://www.shadertoy.com/view/stc3Ws)
        vec3 w2 = w * w;
        vec3 w4 = w2 * w2;
        float k1 = w2.x * w2.z;
        float k2 = w2.x + w2.z;
        float k3 = w4.x + w4.z + w2.y * (w2.y - 6.0 * k2) + 2.0 * k1;
        float k4 = k2 * k2 * k2;
        float k5 = k3 * inversesqrt(k4 * k4 * k2);
        float k6 = w.y * (k2 - w2.y);
        w.x = pos.x + 64.0 * k6 * k5 * w.x * w.z * (w2.x - w2.z) * (w4.x - 6.0 * k1 + w4.z);
        w.y = pos.y - 16.0 * k6 * k6 * k2 + k3 * k3;
        w.z = pos.z - 8.0 * k6 * k5 * (w4.x * (w4.x - 28.0 * k1 + 70.0 * w4.z) + w4.z * (w4.z - 28.0 * k1));

        // - [Inigo Quilez mandlebulb](https://iquilezles.org/articles/mandelbulb/)
        // // extract polar coordinates
        // float wr = sqrt(dot(w, w));
        // float wo = acos(w.y / wr);
        // float wi = atan(w.x, w.z);
        // // scale and rotate the point
        // wr = pow(wr, 8.0);
        // wo = wo * 8.0;
        // wi = wi * 8.0;
        // // convert back to cartesian coordinates
        // w.x = pos.x + wr * sin(wo) * sin(wi);
        // w.y = pos.y + wr * cos(wo);
        // w.z = pos.z + wr * sin(wo) * cos(wi);

        dr = dr * pow(r2, 3.5) * 8.0 + 1.0;
        r2 = dot(w, w);
    }
    
    // distance estimation (through the Hubbard-Douady potential)
    return 0.25 * log(r2) * sqrt(r2) / dr;
}

vec4 march(vec3 ro, vec3 rd) {
    float t = 0.0;
    for (int i = 0; i<ubo.march_iterations && t<ubo.t_max; i++) {
        vec3 pos = ro + t*rd;
        float dt = bulb(pos);
        if (dt < ubo.dt_min) {
            break;
        }
        t += dt;
    }
    return vec4(0.1/t, 0.0, 0.0, 1.0);
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
