#version 460

struct Mouse {
    int x;
    int y;
    uint left;
    uint right;
};

struct Uniforms {
    mat4 world_to_screen;
    vec4 eye;
    Mouse mouse;
    vec4 background_color;
    uint frame;
    float time;
    float deltatime;
    uint width;
    uint height;
    uint monitor_width;
    uint monitor_height;
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

ivec2 to2D(int id, int side) {
    ivec2 pos = ivec2(id % side, (id / side)%side);
    return pos;
}

ivec3 to3D(int id, int side) {
    ivec3 pos = ivec3(id % side, (id / side)%side, (id / (side * side))%side);
    return pos;
}

#ifdef EYEFACE_COMPUTE
    layout(set = 0, binding = 1, rgba16) writeonly uniform image2D screen;
#endif // EYEFACE_COMPUTE

#ifdef EYEFACE_DRAW
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y) * int(gl_WorkGroupSize.x) + int(gl_WorkGroupID.x) * 64;

        if (ubo.width * ubo.height <= id) {
            return;
        }
        ivec2 ipos = to2D(id, int(ubo.width));
        vec2 pos = vec2(ipos);
        vec4 f_color = vec4(0.0);

        pos /= vec2(float(ubo.width), float(ubo.height));
        f_color = vec4(pos.x * pos.x, pos.y, 0.0, 1.0);

        imageStore(screen, ipos, f_color);
    }
#endif // EYEFACE_DRAW
