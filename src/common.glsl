
struct Mouse {
    int x;
    int y;
    uint left;
    uint right;
};

// #define global_id int(gl_LocalInvocationID.x +\
//         gl_LocalInvocationID.y * gl_WorkGroupSize.x +\
//         gl_WorkGroupID.x * gl_WorkGroupSize.x * gl_WorkGroupSize.y)

#define global_id int(gl_GlobalInvocationID.x +\
        gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x +\
        gl_GlobalInvocationID.z * gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupSize.x * gl_WorkGroupSize.y)

uint rand_xorshift(uint state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

uint hash(uint i) {
    i *= 0xB5297A4Du;
    i ^= i >> 8;
    i += 0x68E31DA4u;
    i ^= i << 8;
    i *= 0x1B56C4E9u;
    i ^= i >> 8;
    return i;
}

float fhash(uint i) {
    return float(hash(i))/4294967295.0;
}

uint seed = 0;
float random() {
    return fhash(seed++);
}

vec3 random_normal() {
    vec2 r = vec2(6.28318530718 * random(), acos(2.0 * random() - 1.0));
    vec2 c = cos(r), s = sin(r);
    return vec3(s.y * s.x, s.y * c.x, c.y);
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

