#version 460

struct Mouse {
    int x;
    int y;
    uint left;
    uint right;
};

layout(set = 0, binding = 0) uniform Ubo {
    mat4 transforms[5];
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 world_to_screen;
    vec4 eye;
    Mouse mouse;
    float pitch;
    float yaw;
    uint frame;
} ubo;

struct Vertex {
    vec4 pos;
    vec4 color;
};

layout(location = 0) in vec3 i_pos;

layout(location = 0) out vec3 o_color;

const vec4 p1 = vec4(0.0, -0.5, 0.0, 1.0);
const vec4 p2 = vec4(0.5, 0.5, 0.0, 1.0);
const vec4 p3 = vec4(-0.5, 0.5, 0.0, 1.0);

void main() {
    vec4 selectedPoint;
    if (gl_InstanceIndex == 0) {
        selectedPoint = p1;
    } else if (gl_InstanceIndex == 1) {
        selectedPoint = p2;
    } else {
        selectedPoint = p3;
    }

    vec4 pos = mix(vec4(i_pos, 1.0), selectedPoint, 0.5);

    gl_Position = ubo.world_to_screen * pos;
    o_color = vec3(1.0);
}
