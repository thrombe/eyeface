#version 460

struct Mouse {
    int x;
    int y;
    uint left;
    uint right;
};

layout(set = 0, binding = 0) uniform Ubo {
    mat4 transforms[5];
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 world_to_screen;
    vec4 eye;
    Mouse mouse;
    float pitch;
    float yaw;
    uint frame;
} ubo;

struct Vertex {
    vec4 pos;
};

layout(set = 0, binding = 1) buffer VertexInput {
    Vertex vertices[];
};

layout (local_size_x = 8, local_size_y = 8) in;

uint rand_xorshift(uint state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

void main() {
    uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
    // uint id = gl_GlobalInvocationID.x;
    vec4 pos = vertices[id].pos;

    uint seed = rand_xorshift(id + ubo.frame * 10000000);
    pos = ubo.transforms[seed % 5] * pos;

    vertices[id].pos = pos;
}
