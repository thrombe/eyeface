#version 460

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
    vec4 occlusion_color;
    vec4 sparse_color;
    float occlusion_multiplier;
    float occlusion_attenuation;
    float _pad1;
    float _pad2;
    vec4 voxel_grid_center;
    float voxel_grid_half_size;
    int voxel_grid_side;
    uint frame;
    float time;
} ubo;

struct Vertex {
    vec4 pos;
    vec4 color;
};

layout(location = 0) in vec3 i_pos;

layout(location = 0) out vec3 o_pos;

void main() {
    vec4 pos = vec4(i_pos, 1.0);

    pos = ubo.transforms[gl_InstanceIndex] * pos;
    o_pos = pos.xyz;

    pos = ubo.world_to_screen * pos;
    gl_Position = pos;
}
