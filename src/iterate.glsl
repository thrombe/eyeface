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


bool inGrid(ivec3 pos) {
    if (any(lessThan(pos, ivec3(0)))) {
        return false;
    }
    if (any(greaterThan(pos, ivec3(ubo.voxel_grid_side)))) {
        return false;
    }
    
    return true;
}

int to1D(ivec3 pos, int size) {
    return pos.x + pos.y * size + pos.z * size * size;
}

void main() {
    uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
    // uint id = gl_GlobalInvocationID.x;
    vec4 pos = vertices[id].pos;

    uint seed = rand_xorshift(id + ubo.frame * 10000000);
    pos = ubo.transforms[seed % 5] * pos;

    vertices[id].pos = pos;

    int side = ubo.voxel_grid_side;
    pos.xyz -= ubo.voxel_grid_center.xyz;
    pos /= ubo.voxel_grid_half_size;
    pos *= float(side);
    pos += float(side)/2.0;
    if (inGrid(ivec3(pos))) {
        voxels[to1D(ivec3(pos), side)] += 1;
    }
}
