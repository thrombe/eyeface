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
    vec4 voxel_grid_center;
    int voxel_grid_side;
    float voxel_grid_compensation_perc;
    float occlusion_multiplier;
    float occlusion_attenuation;
    float depth_range;
    float depth_offset;
    float depth_attenuation;
    int points;
    int iterations;
    int voxelization_points;
    int voxelization_iterations;
    int reduction_points;
    uint frame;
    float time;
    float deltatime;
    float lambda;
    float visual_scale;
    float visual_transform_lambda;
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

struct PixelMeta {
    vec3 grid_pos;
    int pix_type;
    vec3 visual_pos;
    float pad;
};

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
        PixelMeta screen[];
    };
    layout(set = 0, binding = 5) buffer DepthBuffer {
        float depth[];
    };
    layout(set = 0, binding = 6) buffer VoxelMetadataBuffer {
        vec4 voxel_grid_min;
        vec4 voxel_grid_max;
        vec4 voxel_grid_mid;
        vec3 reduction_buf[];
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
        PixelMeta screen[];
    };
    layout(set = 0, binding = 4) readonly buffer DepthBuffer {
        float depth[];
    };
    layout(set = 0, binding = 5) readonly buffer VoxelMetadataBuffer {
        vec4 voxel_grid_min;
        vec4 voxel_grid_max;
        vec4 voxel_grid_mid;
        vec3 reduction_buf[];
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

        uint side = ubo.voxel_grid_side;
        if (id < side * side * side) {
            voxels[id] = 0;
            occlusion[id] = 0.0;
        }
        if (id < ubo.width * ubo.height) {
            depth[id] = 1.1;
            screen[id].grid_pos = vec3(0.0);
            screen[id].pix_type = 0;
            screen[id].visual_pos = vec3(0.0);
            screen[id].pad = 0.0;
        }
    }
#endif // EYEFACE_CLEAR_BUFS

#ifdef EYEFACE_ITERATE
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
        vec4 pos = vertices[id].pos;

        uint seed = rand_xorshift(id + ubo.frame * 11335474);
        pos = ubo.transforms[seed % 5] * pos;
        vertices[id].pos = pos;
        if (id < ubo.reduction_points) {
            reduction_buf[id] = pos.xyz;
        }
    }
#endif // EYEFACE_ITERATE

#ifdef EYEFACE_REDUCE
    shared vec4 group_buf[128];

    vec4 reduce(vec4 a, vec4 b) {
        #ifdef EYEFACE_REDUCE_MIN
            return min(a, b);
        #endif
        #ifdef EYEFACE_REDUCE_MAX
            return max(a, b);
        #endif
    }

    void set_reduction_result(vec3 res) {
        #ifdef EYEFACE_REDUCE_MIN
            vec4 old = voxel_grid_min;
        #endif
        #ifdef EYEFACE_REDUCE_MAX
            vec4 old = voxel_grid_max;
        #endif

        vec4 new = vec4(res, 0.0);

        float old_size = length(old.xyz - voxel_grid_mid.xyz);
        float new_size = length(new.xyz - voxel_grid_mid.xyz);
        vec3 compensate = vec3(new_size) * ubo.voxel_grid_compensation_perc;

        #ifdef EYEFACE_REDUCE_MIN
            new -= vec4(compensate, 0.0);
        #endif
        #ifdef EYEFACE_REDUCE_MAX
            new += vec4(compensate, 0.0);
        #endif

        // float e = 0.0;
        // if (old_size < new_size) {
        //     e = 1.0 - exp(-16.0 * ubo.deltatime);
        // } else {
        //     e = 1.0 - exp(-0.01 * ubo.deltatime);
        // }
        // e = 1.0 - exp(-20.0 * ubo.deltatime);
        // new = mix(old, new, e);

        #ifdef EYEFACE_REDUCE_MIN
            voxel_grid_min = new;
        #endif
        #ifdef EYEFACE_REDUCE_MAX
            voxel_grid_max = new;

            vec3 mid = (new.xyz + voxel_grid_min.xyz)/2.0;
            float size = length(new.xyz - voxel_grid_min.xyz)/2.0;
            float e = 1.0 - exp(-ubo.visual_transform_lambda * ubo.deltatime);
            voxel_grid_mid.xyz = mix(voxel_grid_mid.xyz, mid, e);

            size /= ubo.visual_scale;
            voxel_grid_mid.w = mix(voxel_grid_mid.w, size, e);

            if (voxel_grid_mid.w < 0.001) {
                voxel_grid_mid.w = 1.0;
            }
        #endif
    }

    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * 8;
        uint offset = gl_WorkGroupID.x * 64;
        uint gid = id + offset;

        uint index1 = min(0 + gid * 4, ubo.reduction_points - 1);
        uint index2 = min(1 + gid * 4, ubo.reduction_points - 1);
        uint index3 = min(2 + gid * 4, ubo.reduction_points - 1);
        uint index4 = min(3 + gid * 4, ubo.reduction_points - 1);
        vec4 a = vec4(reduction_buf[index1].xyz, 0.0);
        vec4 b = vec4(reduction_buf[index2].xyz, 0.0);
        vec4 c = vec4(reduction_buf[index3].xyz, 0.0);
        vec4 d = vec4(reduction_buf[index4].xyz, 0.0);
        group_buf[id] = reduce(a, b);
        group_buf[id + 64] = reduce(c, d);

        barrier();

        for (int i=64; i>0; i>>=1) {
            if (id < i) {
                vec4 a = group_buf[id];
                vec4 b = group_buf[id + i];
                group_buf[id] = reduce(a, b);
            }
            barrier();
        }

        if (id == 0) {
            reduction_buf[gl_WorkGroupID.x] = group_buf[0].xyz;
            if (gl_NumWorkGroups.x == 1) {
                set_reduction_result(group_buf[0].xyz);
            }
        }
    }
#endif // EYEFACE_REDUCE

#ifdef EYEFACE_PROJECT
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
        // uint id = gl_GlobalInvocationID.x;
        vec4 pos = vertices[id].pos;

        uint seed = rand_xorshift(id + ubo.frame * 13324848);
        for (int i=0; i<ubo.iterations; i++) {
            seed = rand_xorshift(seed);
            pos = ubo.transforms[seed % 5] * pos;

            if (i < ubo.voxelization_iterations && id < ubo.voxelization_points) {
                int side = ubo.voxel_grid_side;
                vec3 grid_pos = pos.xyz;
                grid_pos.xyz -= ubo.voxel_grid_center.xyz + (voxel_grid_min.xyz + voxel_grid_max.xyz)/2.0;
                grid_pos /= voxel_grid_max.xyz - voxel_grid_min.xyz;
                grid_pos *= float(side);
                grid_pos += float(side)/2.0;
                if (inGrid(ivec3(grid_pos))) {
                    voxels[to1D(ivec3(grid_pos), side)] += 1;
                }
            }

            vec4 screen_pos = pos;
            screen_pos.xyz -= voxel_grid_mid.xyz;
            screen_pos.xyz /= voxel_grid_mid.w;
            vec4 visual_pos = screen_pos;
            screen_pos = ubo.world_to_screen * screen_pos;

            // behind the camera
            if (screen_pos.z < 0.0) {
                continue;
            }
            screen_pos /= screen_pos.w;

            // outside the screen
            if (any(greaterThan(abs(screen_pos.xy) - vec2(1.0), vec2(0.0)))) {
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
            grid_pos.xyz -= ubo.voxel_grid_center.xyz + (voxel_grid_min.xyz + voxel_grid_max.xyz)/2.0;
            grid_pos /= voxel_grid_max.xyz - voxel_grid_min.xyz;
            grid_pos *= float(side);
            grid_pos += float(side)/2.0;
            screen[si].grid_pos = grid_pos;
            screen[si].visual_pos = visual_pos.xyz;
            if (inGrid(ivec3(grid_pos))) {
                screen[si].pix_type = 2;
            } else {
                screen[si].pix_type = 1;
            }
        }

        // vertices[id].pos = pos;
    }
#endif // EYEFACE_PROJECT

#ifdef EYEFACE_OCCLUSION
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y) * int(gl_WorkGroupSize.x) + int(gl_WorkGroupID.x) * 64;
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
        occlusion[id] = o;
    }
#endif // EYEFACE_OCCLUSION

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

        vec3 vpos = screen[index].visual_pos;
        float dist = length(vpos - ubo.eye.xyz);
        // https://www.desmos.com/calculator/ted75acgr5
        dist = 1.0/(1.0 + exp(-pow(clamp(dist - ubo.depth_offset, 0.0, 30.0), ubo.depth_attenuation) * 6.5 / ubo.depth_range + 3.5));

        int type = screen[index].pix_type;
        if (type == 2) {
            vec3 pos = screen[index].grid_pos;
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
            value = pow(max(value, 0.0)*ubo.occlusion_multiplier, ubo.occlusion_attenuation);

            f_color = vec4(mix(ubo.occlusion_color.xyz, ubo.sparse_color.xyz, mix(value, 0.0, dist)), 1.0);
        } else if (type == 1) {
            f_color = vec4(mix(ubo.sparse_color.xyz, ubo.occlusion_color.xyz, dist), 1.0);
        } else {
            f_color = ubo.background_color;
        }

        float gamma = 2.1;
        f_color = vec4(
            pow(f_color.x, gamma),
            pow(f_color.y, gamma),
            pow(f_color.z, gamma),
            1.0
        );
    }
#endif // EYEFACE_FRAG
