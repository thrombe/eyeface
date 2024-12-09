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

int to1D(ivec3 pos, int size) {
    return pos.x + pos.y * size + pos.z * size * size;
}
ivec3 to3D(int id, int side) {
    ivec3 pos = ivec3(id % side, (id / side)%side, (id / (side * side))%side);
    return pos;
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

float voxelGridSample(ivec3 pos) {
    if (!inGrid(pos)) {
        return 0;
    }

    return float(voxels[to1D(pos, ubo.voxel_grid_side)] > 0);
}

void main() {
    int id = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y) * int(gl_WorkGroupSize.x) + int(gl_WorkGroupID.x) * 64;
    // uint id = gl_GlobalInvocationID.x;
    // vec4 pos = vertices[id].pos;
    int side = ubo.voxel_grid_side;
    ivec3 pos = to3D(id, side);
    for (int z = -1; z < 2; z += 1) {
        for (int y = -1; y < 2; y += 1) {
            for (int x = -1; x < 2; x += 1) {
                occlusion[id] += voxelGridSample(pos + ivec3(x, y, z));
            }
        }
    }
    occlusion[id] /= 27.0;

    // uint seed = rand_xorshift(id + ubo.frame * 10000000);
    // pos = ubo.transforms[seed % 5] * pos;

    // vertices[id].pos = pos;
}
