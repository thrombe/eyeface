#version 460

struct Mouse {
    int x;
    int y;
    uint left;
    uint right;
};

struct Uniforms {
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
};

struct Vertex {
    vec4 pos;
};

layout(set = 0, binding = 0) uniform Ubo {
    Uniforms ubo;
};

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

#ifdef EYEFACE_COMPUTE
    layout(set = 0, binding = 1) buffer VertexInput {
        Vertex vertices[];
    };

    layout(set = 0, binding = 2) buffer VoxelBuffer {
        uint voxels[];
    };

    layout(set = 0, binding = 3) buffer OcclusionBuffer {
        float occlusion[];
    };
#endif // EYEFACE_COMPUTE

#ifdef EYEFACE_RENDER
    layout(set = 0, binding = 1) readonly buffer VoxelBuffer {
        uint voxels[];
    };
    layout(set = 0, binding = 2) readonly buffer OcclusionBuffer {
        float occlusion[];
    };
#endif // EYEFACE_RENDER

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

#ifdef EYEFACE_CLEAR_BUFS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;

        voxels[id] = 0;
        occlusion[id] = 0;
    }
#endif // EYEFACE_CLEAR_BUFS

#ifdef EYEFACE_ITERATE
    layout (local_size_x = 8, local_size_y = 8) in;
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
#endif // EYEFACE_ITERATE

#ifdef EYEFACE_OCCLUSION
    layout (local_size_x = 8, local_size_y = 8) in;
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
#endif // EYEFACE_OCCLUSION

#ifdef EYEFACE_VERT
    layout(location = 0) in vec3 i_pos;
    layout(location = 0) out vec3 o_pos;

    void main() {
        vec4 pos = vec4(i_pos, 1.0);

        pos = ubo.transforms[gl_InstanceIndex] * pos;
        o_pos = pos.xyz;

        pos = ubo.world_to_screen * pos;
        gl_Position = pos;
    }
#endif // EYEFACE_VERT

#ifdef EYEFACE_FRAG
    layout(location = 0) in vec3 v_pos;
    layout(location = 0) out vec4 f_color;

    void main() {
        float d = length(v_pos - ubo.eye.xyz);
        d = 1.0/d;
        d = d * d;

        int side = ubo.voxel_grid_side;
        vec3 pos = v_pos;
        pos -= ubo.voxel_grid_center.xyz;
        pos /= ubo.voxel_grid_half_size;
        pos *= float(side);
        pos += float(side)/2.0;
        int index = to1D(ivec3(pos), side);

        float o = 1.0 - occlusion[index];
    	o = pow(clamp(o * ubo.occlusion_multiplier, 0.0, 1.0), ubo.occlusion_attenuation);

    	if (inGrid(ivec3(pos))) {
        	f_color = vec4(mix(ubo.occlusion_color.xyz, ubo.sparse_color.xyz, o), 1);
    	} else {
    	    f_color = vec4(vec3(d), 1);
    	}
    }
#endif // EYEFACE_FRAG
