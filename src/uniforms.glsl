 // This file is generated from code. DO NOT EDIT.

 struct Mouse {
     int x;
     int y;
     uint left;
     uint right;
 };

 struct Uniforms {
     mat4 transforms[5];
     mat4 world_to_screen;
     vec4 eye;
     Mouse mouse;
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
     uint frame;
     float time;
     float deltatime;
     float lambda;
     float visual_scale;
     float visual_transform_lambda;
     uint width;
     uint height;
     uint monitor_width;
     uint monitor_height;
 };

