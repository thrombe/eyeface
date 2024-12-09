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

layout(set = 0, binding = 1) readonly buffer VoxelBuffer {
    uint voxels[];
};
layout(set = 0, binding = 2) readonly buffer OcclusionBuffer {
    float occlusion[];
};
layout(location = 0) in vec3 v_pos;

layout(location = 0) out vec4 f_color;

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
