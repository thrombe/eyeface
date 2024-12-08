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
    uint frame;
    float time;
} ubo;

layout(location = 0) in vec3 v_pos;

layout(location = 0) out vec4 f_color;

void main() {
    float d = length(v_pos - ubo.eye.xyz);
    d = 1.0/d;
    d = d * d;
    f_color = vec4(vec3(d), 1.0);
}
