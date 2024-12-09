#version 450

struct Mouse {
    int x;
    int y;
    uint left;
    uint right;
};

layout(set = 0, binding = 0) uniform Ubo {
    mat4 transforms[5];
    mat4 world_to_screen;
    vec4 eye;
    Mouse mouse;
    uint frame;
    float time;
} ubo;

layout(set = 0, binding = 1) readonly buffer VoxelBuffer {
    uint voxels[];
};
layout(set = 0, binding = 2) readonly buffer OcclusionBuffer {
    float occlusion[];
};
layout(location = 0) in vec3 v_pos;

layout(location = 0) out vec4 f_color;

void main() {
    float d = length(v_pos - ubo.eye.xyz);
    d = 1.0/d;
    d = d * d;

    vec3 pos = v_pos;
    pos += 1.5;
    pos *= 100.0;
    pos /= 3.0;
    uint index = uint(pos.z) * 300 * 300 + uint(pos.y) * 300 + uint(pos.x);

    float o = 1.0 - occlusion[index];
	o = pow(clamp(o * 1.0, 0.0, 1.0), 2.0);

	vec3 col2 = vec3(0.7, 0.8, 0.6);
	vec3 col1 = vec3(0.2, 0.0, 0.0);
	f_color = vec4(mix(col1, col2, o), 1);
}
