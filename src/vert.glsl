#version 460

layout(set = 0, binding = 0) uniform Ubo {
    float a;
    float b;
} ubo;

struct Vertex {
    vec4 pos;
    vec4 color;
};

layout(std140, set = 0, binding = 1) buffer idk {
    vec2 vertices[];
};

// layout(location = 0) in vec2 b_pos;
// layout(location = 1) in vec3 a_color;

layout(location = 0) out vec3 v_color;

void main() {
    // ssbo.vertices[0] = vec2(1.0);
    vec2 a_pos = vertices[gl_VertexIndex].xy;
    // a_pos = vec2(float(gl_VertexIndex) / 10000);
    // vertices[0] = vec4(a_pos, 0.0, 0.0);
    // a_pos = vertices[0].xy;
    gl_Position = vec4(a_pos, 0.0, 1.0);
    // gl_Position = vec4(b_pos, 0.0, 1.0);
    v_color = vec3(1.0);
}
