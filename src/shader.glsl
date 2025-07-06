#version 460

#include <common.glsl>
#include <uniforms.glsl>

#ifdef COMPUTE_PASS
    #define bufffer buffer
#else
    #define bufffer readonly buffer
#endif

layout(set = _set_compute, binding = _bind_uniforms) uniform Ubo {
    Uniforms ubo;
};

struct PixelMeta {
    vec3 grid_pos;
    int pix_type;
    vec3 visual_pos;
    float pad;
};

#ifdef COMPUTE_PASS
    layout(set = _set_compute, binding = _bind_points) bufffer PointsBuffer {
        vec4 points[];
    };
    layout(set = _set_compute, binding = _bind_voxels) bufffer VoxelBuffer {
        uint voxels[];
    };
    layout(set = _set_compute, binding = _bind_occlusion) bufffer OcclusionBuffer {
        float occlusion[];
    };
    layout(set = _set_compute, binding = _bind_gbuffer) bufffer GBuffer {
        PixelMeta gbuffer[];
    };
    layout(set = _set_compute, binding = _bind_screen, rgba16f) writeonly uniform image2D screen;
    // keeping a separate depth buffer is faster cuz the access patterns of this are much different
    layout(set = _set_compute, binding = _bind_screen_depth) bufffer DepthBuffer {
        float depth[];
    };
    layout(set = _set_compute, binding = _bind_reduction) bufffer ReductionBuffer {
        vec4 voxel_grid_min;
        vec4 voxel_grid_max;
        vec4 voxel_grid_mid;
        vec3 reduction_buf[];
    };
#endif // COMPUTE_PASS

layout(push_constant) uniform PushConstants_ {
    PushConstants push;
};

void set_seed(int id) {
    seed = int(ubo.frame.frame) ^ id ^ floatBitsToInt(ubo.frame.time) ^ push.seed;
}

bool inGrid(ivec3 pos) {
    if (any(lessThan(pos, ivec3(0)))) {
        return false;
    }
    if (any(greaterThan(pos, ivec3(ubo.params.voxel_grid_side)))) {
        return false;
    }

    return true;
}

float voxelGridSample(ivec3 pos) {
    if (!inGrid(pos)) {
        return 0;
    }
    return float(voxels[to1D(pos, ubo.params.voxel_grid_side)] > 0);
}

#ifdef CLEAR_BUFS_PASS
    layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        int id = global_id;

        uint side = ubo.params.voxel_grid_side;
        if (id < side * side * side) {
            voxels[id] = 0;
            occlusion[id] = 0.0;
        }
        if (id < ubo.frame.width * ubo.frame.height) {
            depth[id] = 1.1;
            gbuffer[id].grid_pos = vec3(0.0);
            gbuffer[id].pix_type = 0;
            gbuffer[id].visual_pos = vec3(0.0);
            gbuffer[id].pad = 0.0;
            imageStore(screen, to2D(int(id), int(ubo.frame.width)), ubo.params.background_color);
        }
    }
#endif // CLEAR_BUFS_PASS

#ifdef ITERATE_PASS
    layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        int id = global_id;
        vec4 pos = points[id];

        set_seed(id);
        uint t_index = rand_xorshift(uint(id) + uint(ubo.frame.frame) * 11335474u + uint(push.seed)) % 5;
        pos = ubo.transforms[t_index] * pos;
        points[id] = pos;
        
        if (id < ubo.params.reduction_points) {
            reduction_buf[id] = pos.xyz;
        }
    }
#endif // ITERATE_PASS

#ifdef REDUCE_PASS
    shared vec4 group_buf[128];

    vec4 reduce_op(vec4 a, vec4 b) {
        #ifdef REDUCE_MIN_PASS
            return min(a, b);
        #endif
        #ifdef REDUCE_MAX_PASS
            return max(a, b);
        #endif
    }

    void set_reduction_result(vec3 res) {
        #ifdef REDUCE_MIN_PASS
            vec4 old = voxel_grid_min;
        #endif
        #ifdef REDUCE_MAX_PASS
            vec4 old = voxel_grid_max;
        #endif

        vec4 new = vec4(res, 0.0);

        float old_size = length(old.xyz - voxel_grid_mid.xyz);
        float new_size = length(new.xyz - voxel_grid_mid.xyz);
        vec3 compensate = vec3(new_size) * ubo.params.voxel_grid_compensation_perc;

        #ifdef REDUCE_MIN_PASS
            new -= vec4(compensate, 0.0);
        #endif
        #ifdef REDUCE_MAX_PASS
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

        #ifdef REDUCE_MIN_PASS
            voxel_grid_min = new;
        #endif
        #ifdef REDUCE_MAX_PASS
            voxel_grid_max = new;

            vec3 mid = (new.xyz + voxel_grid_min.xyz)/2.0;
            float size = length(new.xyz - voxel_grid_min.xyz)/2.0;
            float e = 1.0 - exp(-ubo.params.visual_transform_lambda * ubo.frame.deltatime);
            voxel_grid_mid.xyz = mix(voxel_grid_mid.xyz, mid, e);

            size /= ubo.params.visual_scale;
            voxel_grid_mid.w = mix(voxel_grid_mid.w, size, e);

            if (voxel_grid_mid.w < 0.001) {
                voxel_grid_mid.w = 1.0;
            }
        #endif
    }

    layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * 8;
        uint offset = gl_WorkGroupID.x * 64;
        uint gid = id + offset;

        uint index1 = min(0 + gid * 4, ubo.params.reduction_points - 1);
        uint index2 = min(1 + gid * 4, ubo.params.reduction_points - 1);
        uint index3 = min(2 + gid * 4, ubo.params.reduction_points - 1);
        uint index4 = min(3 + gid * 4, ubo.params.reduction_points - 1);
        vec4 a = vec4(reduction_buf[index1].xyz, 0.0);
        vec4 b = vec4(reduction_buf[index2].xyz, 0.0);
        vec4 c = vec4(reduction_buf[index3].xyz, 0.0);
        vec4 d = vec4(reduction_buf[index4].xyz, 0.0);
        group_buf[id] = reduce_op(a, b);
        group_buf[id + 64] = reduce_op(c, d);

        barrier();

        for (int i=64; i>0; i>>=1) {
            if (id < i) {
                vec4 val1 = group_buf[id];
                vec4 val2 = group_buf[id + i];
                group_buf[id] = reduce_op(val1, val2);
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
#endif // REDUCE_PASS

#ifdef PROJECT_PASS
    layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        int id = global_id;
        vec4 pos = points[id];

        set_seed(id);
        uint seed = rand_xorshift(uint(id) + uint(ubo.frame.frame) * 13324848u + uint(push.seed));
        for (int i=0; i<ubo.params.iterations; i++) {
            seed = rand_xorshift(seed);
            pos = ubo.transforms[seed % 5] * pos;

            if (i < ubo.params.voxelization_iterations && id < ubo.params.voxelization_points) {
                int side = ubo.params.voxel_grid_side;
                vec3 grid_pos = pos.xyz;
                grid_pos.xyz -= ubo.params.voxel_grid_center.xyz + (voxel_grid_min.xyz + voxel_grid_max.xyz)/2.0;
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

            screen_pos = ubo.params.world_to_screen * screen_pos;

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
            screen_pos.xy = screen_pos.xy * 0.5 + 0.5;
            ivec2 screen_icoord = ivec2(screen_pos.xy * vec2(float(ubo.frame.width), float(ubo.frame.height)));
            int si = to1D(screen_icoord, int(ubo.frame.width));

            // NOTE: there's many race conditions here, but since (screen resolution) >> (number of hits at the same
            //    position and at the same time) we can get away with ignoring the race conditions
            if (depth[si] < screen_pos.z) {
                continue;
            }
            depth[si] = screen_pos.z;

            int side = ubo.params.voxel_grid_side;
            vec3 grid_pos = pos.xyz;
            grid_pos.xyz -= ubo.params.voxel_grid_center.xyz + (voxel_grid_min.xyz + voxel_grid_max.xyz)/2.0;
            grid_pos /= voxel_grid_max.xyz - voxel_grid_min.xyz;
            grid_pos *= float(side);
            grid_pos += float(side)/2.0;
            // TODO: maybe just store global pos and derive the rest

            gbuffer[si].grid_pos = grid_pos;
            gbuffer[si].visual_pos = visual_pos.xyz;
            if (inGrid(ivec3(grid_pos))) {
                gbuffer[si].pix_type = 2;
            } else {
                gbuffer[si].pix_type = 1;
            }
        }
    }
#endif // PROJECT_PASS

#ifdef OCCLUSION_PASS
    layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        int id = global_id;
        int side = ubo.params.voxel_grid_side;
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
#endif // OCCLUSION_PASS

#ifdef DRAW_PASS
    layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
    void main() {
        int id = global_id;

        if (ubo.frame.width * ubo.frame.height <= id) {
            return;
        }

        vec3 f_color = vec3(0.0);

        vec3 vpos = gbuffer[id].visual_pos;
        float dist = length(vpos - ubo.camera.eye.xyz);
        // https://www.desmos.com/calculator/ted75acgr5
        dist = 1.0/(1.0 + exp(-pow(clamp(dist - ubo.params.depth_offset, 0.0, 30.0), ubo.params.depth_attenuation) * 6.5 / ubo.params.depth_range + 3.5));

        int type = gbuffer[id].pix_type;
        if (type == 2) {
            vec3 pos = gbuffer[id].grid_pos;
            int side = ubo.params.voxel_grid_side;

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
            value = pow(max(value, 0.0)*ubo.params.occlusion_multiplier, ubo.params.occlusion_attenuation);

            f_color = vec3(mix(ubo.params.occlusion_color.xyz, ubo.params.sparse_color.xyz, mix(value, 0.0, dist)));
        } else if (type == 1) {
            f_color = vec3(mix(ubo.params.sparse_color.xyz, ubo.params.occlusion_color.xyz, dist));
        } else {
            f_color = ubo.params.background_color.xyz;
        }

        imageStore(screen, to2D(id, int(ubo.frame.width)), vec4(f_color, 1.0));
    }
#endif // DRAW_PASS
