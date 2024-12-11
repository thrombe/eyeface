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
    vec4 background_color;
    float occlusion_multiplier;
    float occlusion_attenuation;
    int iterations;
    int voxelization_iterations;
    vec4 voxel_grid_center;
    float voxel_grid_half_size;
    int voxel_grid_side;
    uint frame;
    float time;
    uint width;
    uint height;
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

int to1D(ivec2 pos, int size) {
    return pos.x + pos.y * size;
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
    layout(set = 0, binding = 4) buffer ScreenBuffer {
        vec4 screen[];
    };
    layout(set = 0, binding = 5) buffer DepthBuffer {
        float depth[];
    };
#endif // EYEFACE_COMPUTE

#ifdef EYEFACE_RENDER
    layout(set = 0, binding = 1) readonly buffer VoxelBuffer {
        uint voxels[];
    };
    layout(set = 0, binding = 2) readonly buffer OcclusionBuffer {
        float occlusion[];
    };
    layout(set = 0, binding = 3) readonly buffer ScreenBuffer {
        vec4 screen[];
    };
    layout(set = 0, binding = 4) readonly buffer DepthBuffer {
        float depth[];
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
        occlusion[id] = 0.0;
        depth[id] = 1.1;
        screen[id] = vec4(0.0);
    }
#endif // EYEFACE_CLEAR_BUFS

#ifdef EYEFACE_VOXELIZE
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
        // uint id = gl_GlobalInvocationID.x;
        vec4 pos = vertices[id].pos;

        uint seed = rand_xorshift(id + ubo.frame * 11335474);
        for (int i=0; i<ubo.voxelization_iterations; i++) {
            seed = rand_xorshift(seed);
            pos = ubo.transforms[seed % 5] * pos;

            int side = ubo.voxel_grid_side;
            vec3 grid_pos = pos.xyz;
            grid_pos.xyz -= ubo.voxel_grid_center.xyz;
            grid_pos /= ubo.voxel_grid_half_size;
            grid_pos *= float(side);
            grid_pos += float(side)/2.0;
            if (inGrid(ivec3(grid_pos))) {
                voxels[to1D(ivec3(grid_pos), side)] += 1;
            }
        }

        vertices[id].pos = pos;
    }
#endif // EYEFACE_VOXELIZE

#ifdef EYEFACE_OCCLUSION
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y) * int(gl_WorkGroupSize.x) + int(gl_WorkGroupID.x) * 64;
        // uint id = gl_GlobalInvocationID.x;
        // vec4 pos = vertices[id].pos;
        int side = ubo.voxel_grid_side;
        ivec3 pos = to3D(id, side);
        float o = 0.0;
        for (int z = -1; z < 2; z += 1) {
            for (int y = -1; y < 2; y += 1) {
                for (int x = -1; x < 2; x += 1) {
                    o += voxelGridSample(pos + ivec3(x, y, z));
                }
            }
        }
        o /= 27.0;
        o = 1.0 - o;
        occlusion[id] = pow(clamp(o * ubo.occlusion_multiplier, 0.0, 1.0), ubo.occlusion_attenuation);
    }
#endif // EYEFACE_OCCLUSION

#ifdef EYEFACE_ITERATE
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
        // uint id = gl_GlobalInvocationID.x;
        vec4 pos = vertices[id].pos;

        uint seed = rand_xorshift(id + ubo.frame * 13324848);
        for (int i=0; i<ubo.iterations; i++) {
            seed = rand_xorshift(seed);
            pos = ubo.transforms[seed % 5] * pos;

            vec4 screen_pos = ubo.world_to_screen * pos;
            screen_pos /= screen_pos.w;

            // outside the screen
            if (any(greaterThan(abs(screen_pos.xy) - vec2(1.0), vec2(0.0)))) {
                continue;
            }
            if (screen_pos.z < 0.0) {
                continue;
            }

            // depth testing
            screen_pos.xy /= 2.0;
            screen_pos.xy += 0.5;
            int si = to1D(ivec2(screen_pos.xy * vec2(float(ubo.width), float(ubo.height))), int(ubo.width));

            // NOTE: there's many race conditions here, but since (screen resolution) >> (number of hits at the same
            //    position and at the same time) we can get away with ignoring the race conditions
            // float d = atomicMin(depth[si], screen_pos.z);
            float d = depth[si];
            if (d < screen_pos.z) {
                continue;
            }
            depth[si] = screen_pos.z;

            int side = ubo.voxel_grid_side;
            vec3 grid_pos = pos.xyz;
            grid_pos -= ubo.voxel_grid_center.xyz;
            grid_pos /= ubo.voxel_grid_half_size;
            grid_pos *= float(side);
            grid_pos += float(side)/2.0;
            if (inGrid(ivec3(grid_pos))) {
                screen[si].xyz = grid_pos;
                screen[si].w = 2.0;
            } else {
                screen[si].xyz = pos.xyz;
                screen[si].w = 1.0;
            }
        }

        vertices[id].pos = pos;
    }
#endif // EYEFACE_ITERATE

#ifdef EYEFACE_VERT
    void main() {
        vec3 positions[6] = vec3[6](
            vec3(1.0, 1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(1.0, -1.0, 0.0),
            vec3(-1.0, 1.0, 0.0),
            vec3(-1.0, -1.0, 0.0)
        );

        vec3 pos = positions[gl_VertexIndex];

        gl_Position = vec4(pos, 1.0);
    }
#endif // EYEFACE_VERT

#ifdef EYEFACE_FRAG
    layout(location = 0) out vec4 f_color;

    void main() {
        ivec2 pos = ivec2(gl_FragCoord.xy);
        int index = to1D(pos, int(ubo.width));

        float type = screen[index].w;
        if (type > 1.5) {
            vec3 pos = screen[index].xyz;
            int side = ubo.voxel_grid_side;
            int index = to1D(ivec3(pos), side);

            // trilinear occlusion sample :/
			ivec3 vi = ivec3(pos);
			float weight1 = 0.0f;
			float weight2 = 0.0f;
			float weight3 = 0.0f;
			float value = 0.0f;
			for (int i = 0; i < 2; ++i) {
				weight1 = 1 - min(abs(pos.x - (vi.x + i)), side);
				for (int j = 0; j < 2; ++j) {
					weight2 = 1 - min(abs(pos.y - (vi.y + j)), side);
					for (int k = 0; k < 2; ++k) {
						weight3 = 1 - min(abs(pos.z - (vi.z + k)), side);
						value += weight1 * weight2 * weight3 * occlusion[to1D(vi + ivec3(i, j, k), side)];
					}
				}
			}

            f_color = vec4(mix(ubo.occlusion_color.xyz, ubo.sparse_color.xyz, value), 1.0);
        } else if (type > 0.5) {
            vec3 pos = screen[index].xyz;
            float dist = length(pos - ubo.eye.xyz);
            dist = 1.0/dist;
            dist = dist * dist;

            f_color = vec4(vec3(dist), 1.0);
        } else {
            f_color = ubo.background_color;
        }
    }
#endif // EYEFACE_FRAG
