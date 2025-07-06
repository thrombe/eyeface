 // This file is generated from code. DO NOT EDIT.

 struct Mouse {
     int x;
     int y;
     uint left;
     uint right;
 };

 struct Camera3DMeta {
     uint did_change;
     uint did_move;
     uint did_rotate;
 };

 struct Camera3D {
     vec3 eye;
     vec3 fwd;
     vec3 right;
     vec3 up;
     Camera3DMeta meta;
 };

 struct Frame {
     uint frame;
     float time;
     float deltatime;
     int width;
     int height;
     int monitor_width;
     int monitor_height;
 };

 struct Params {
     mat4 world_to_screen;
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
     float lambda;
     float visual_scale;
     float visual_transform_lambda;
 };

 struct Uniforms {
     mat4 transforms[5];
     Camera3D camera;
     Mouse mouse;
     Frame frame;
     Params params;
 };

 struct PushConstants {
     int seed;
 };

 const int _set_compute = 0;

 const int _bind_uniforms = 0;
 const int _bind_points = 1;
 const int _bind_voxels = 2;
 const int _bind_occlusion = 3;
 const int _bind_gbuffer = 4;
 const int _bind_screen = 5;
 const int _bind_screen_depth = 6;
 const int _bind_reduction = 7;

