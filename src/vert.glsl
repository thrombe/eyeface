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

// layout(std140, set = 0, binding = 1) buffer idk {
//     vec2 vertices[];
// };

layout(location = 0) in vec2 b_pos;
layout(location = 1) in vec3 a_color;

layout(location = 0) out vec3 v_color;

void main() {
    // ssbo.vertices[0] = vec2(1.0);
    // vec2 a_pos = vertices[gl_VertexIndex].xy;
    // a_pos = vec2(float(gl_VertexIndex) / 10000);
    // vertices[0] = vec4(a_pos, 0.0, 0.0);
    // a_pos = vertices[0].xy;
    // gl_Position = vec4(a_pos, 0.0, 1.0);
    vec4 pos = vec4(0.0);
    pos = vec4(b_pos, 3.0, 1.0);
    // gl_Position = ubo.view_matrix * gl_Position;
    // gl_Position = ubo.projection_matrix * gl_Position;
    pos = ubo.world_to_screen * pos;
    // pos = ubo.view_matrix * pos;
    // pos = ubo.projection_matrix * pos;
    // pos = (ubo.projection_matrix * ubo.view_matrix) * pos;
    // pos /= pos.w;
    gl_Position = pos;
    v_color = a_color;
}
