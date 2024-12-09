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
    uint frame;
    float time;
} ubo;

struct Vertex {
    vec4 pos;
};

layout(set = 0, binding = 1) buffer VertexInput {
    Vertex vertices[];
};
layout(set = 0, binding = 2) buffer VoxelBuffer {
    uint voxels[];
};
layout(set = 0, binding = 3) buffer OcclusionBuffer {
    float occlusion[];
};

layout (local_size_x = 8, local_size_y = 8) in;

uint rand_xorshift(uint state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

uint to1D(ivec3 pos, int size) {
    return pos.x + pos.y * size + pos.z * size * size;
}

void main() {
    int id = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y) * int(gl_WorkGroupSize.x) + int(gl_WorkGroupID.x) * 64;
    // uint id = gl_GlobalInvocationID.x;
    // vec4 pos = vertices[id].pos;
    ivec3 pos = ivec3(id % 300, (id / 300)%300, (id / (300 * 300))%300);
    for (int z = -1; z < 2; z += 1) {
        for (int y = -1; y < 2; y += 1) {
            for (int x = -1; x < 2; x += 1) {
                occlusion[id] += float(voxels[to1D(pos + ivec3(x, y, z), 300)] > 0);
            }
        }
    }
    occlusion[id] /= 27.0;

    // uint seed = rand_xorshift(id + ubo.frame * 10000000);
    // pos = ubo.transforms[seed % 5] * pos;

    // vertices[id].pos = pos;
}
