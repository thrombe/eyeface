#version 460

layout(set = 0, binding = 0) uniform Ubo {
    uint frame;
    float b;
} ubo;

struct Vertex {
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 1) buffer VertexInput {
    Vertex vertices[];
};

layout (local_size_x = 8, local_size_y = 8) in;

// layout(rgba16f,set = 0, binding = 0) uniform image2D image;

const vec4 p1 = vec4(0.0, -0.5, 0.0, 0.0);
const vec4 p2 = vec4(0.5, 0.5, 0.0, 0.0);
const vec4 p3 = vec4(-0.5, 0.5, 0.0, 0.0);

// Simple RNG function (XORSHIFT)
uint rand_xorshift(uint state) {
    state ^= (state << 13);
    state ^= (state >> 17);
    state ^= (state << 5);
    return state;
}

void main() {
    uint id = gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_WorkGroupID.x * 64;
    // uint id = gl_GlobalInvocationID.x; // Unique ID for this invocation
    vec4 pos = vertices[id].pos; // Read initial position from vertices[]

    // Perform random selections and averaging
    for (int i = 0; i < 1; ++i) {
        // Update seed for randomness
        uint seed = rand_xorshift(id + ubo.frame * 100000); // Update RNG state
        uint choice = seed % 3; // Randomly choose between p1, p2, p3

        vec4 selectedPoint;
        if (choice == 0) {
            selectedPoint = p1;
        } else if (choice == 1) {
            selectedPoint = p2;
        } else {
            selectedPoint = p3;
        }

        // Update position
        pos = mix(pos, selectedPoint, 0.5);
    }

    // Average the position after the loop
    // pos.x /= 1.0;
    // pos.y /= 1.0;

    // Write back to the vertices buffer
    vertices[id].pos = pos; // Store updated position back in
}

// void main() {
//     // ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);

//     // vec3 pos = gl_GlobalInvocationID.xyz;

//     // uvec3 globalID = gl_GlobalInvocationID; // Unique ID across all invocations
//     uvec3 localID = gl_LocalInvocationID; // ID within the workgroup

//     // Calculate linear index for accessing arrays
//     uint linearIndex = localID.x + localID.y * gl_WorkGroupSize.x;
//     // vertices[linearIndex].pos = vec4(0.0);
//     // vertices[gl_GlobalInvocationID.x].pos = vec4(0.0);
//     // for (uint i = 0; i < 10000; i += 1) {
//     //     vertices[i].pos = vec4(1.0);
//     //     vertices[i].color = vec4(1.0);
//     // }

//     // ivec2 size = imageSize(image);

//     // if (texelCoord.x < size.x && texelCoord.y < size.y) {
//     //     vec4 color = vec4(0.0, 0.0, 0.0, 1.0);

//     //     if(gl_LocalInvocationID.x != 0 && gl_LocalInvocationID.y != 0)
//     //     {
//     //         color.x = float(texelCoord.x)/(size.x);
//     //         color.y = float(texelCoord.y)/(size.y);	
//     //     }
    
//     //     imageStore(image, texelCoord, color);
//     // }
// }
