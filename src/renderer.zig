const std = @import("std");

const vk = @import("vulkan");

const utils = @import("utils.zig");
const Fuse = utils.Fuse;

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Mat4x4 = math.Mat4x4;

const transform = @import("transform.zig");

const Engine = @import("engine.zig");
const c = Engine.c;

const gui = @import("gui.zig");
const GuiEngine = gui.GuiEngine;

const state = @import("state.zig");
const AppState = state.AppState;

const render_utils = @import("render_utils.zig");
const Swapchain = render_utils.Swapchain;
const UniformBuffer = render_utils.UniformBuffer;
const Buffer = render_utils.Buffer;

const main = @import("main.zig");
const allocator = main.allocator;

const Renderer = @This();

swapchain: Swapchain,
uniforms: UniformBuffer(Uniforms),
vertex_buffer: Buffer,
// - [Depth buffering - Vulkan Tutorial](https://vulkan-tutorial.com/Depth_buffering)
// - [Setting up depth buffer - Vulkan Guide](https://vkguide.dev/docs/chapter-3/depth_buffer/)
depth_image: vk.Image,
depth_buffer_memory: vk.DeviceMemory,
depth_image_view: vk.ImageView,
voxel_buffer: Buffer,
occlusion_buffer: Buffer,
screen_buffer: Buffer,
screen_depth_buffer: Buffer,
reduction_buffer: Buffer,
descriptor_pool: vk.DescriptorPool,
frag_descriptor_set_layout: vk.DescriptorSetLayout,
frag_descriptor_set: vk.DescriptorSet,
compute_descriptor_set_layout: vk.DescriptorSetLayout,
compute_descriptor_set: vk.DescriptorSet,
pass: vk.RenderPass,
compute_pipeline_layout: vk.PipelineLayout,
compute_pipelines: []vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
pipeline: vk.Pipeline,
// framebuffers are objects containing views of swapchain images
framebuffers: []vk.Framebuffer,
command_pool: vk.CommandPool,
// command buffers are recordings of a bunch of commands that a gpu can execute
command_buffers: []vk.CommandBuffer,

const Device = Engine.VulkanContext.Api.Device;
pub const Vertex = extern struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .input_rate = .vertex,
    };
    const attribute_description = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r32g32b32_sfloat,
            .offset = @offsetOf(Vertex, "pos"),
        },
    };

    pos: [4]f32,
};

pub const Uniforms = extern struct {
    transforms: TransformSet,
    world_to_screen: Mat4x4,
    eye: Vec4,
    mouse: extern struct { x: i32, y: i32, left: u32, right: u32 },
    occlusion_color: Vec4,
    sparse_color: Vec4,
    background_color: Vec4,
    voxel_grid_center: Vec4,
    voxel_grid_side: u32,
    voxel_grid_compensation_perc: f32,
    occlusion_multiplier: f32,
    occlusion_attenuation: f32,
    depth_range: f32,
    depth_offset: f32,
    depth_attenuation: f32,
    points: u32,
    iterations: u32,
    voxelization_points: u32,
    voxelization_iterations: u32,
    reduction_points: u32,
    frame: u32,
    time: f32,
    deltatime: f32,
    lambda: f32,
    visual_scale: f32,
    visual_transform_lambda: f32,
    width: u32,
    height: u32,

    pub const TransformSet = transform.TransformSet(5);
};

pub fn init(engine: *Engine, app_state: *AppState) !@This() {
    var ctx = &engine.graphics;
    const device = &ctx.device;

    var swapchain = try Swapchain.init(ctx, engine.window.extent);
    errdefer swapchain.deinit(device);

    var uniforms = try UniformBuffer(Uniforms).new(app_state.uniforms(engine.window), ctx);
    errdefer uniforms.deinit(device);

    const pool = try device.createCommandPool(&.{
        .queue_family_index = ctx.graphics_queue.family,
        .flags = .{
            .reset_command_buffer_bit = true,
        },
    }, null);
    errdefer device.destroyCommandPool(pool, null);

    var vertex_buffer = try Buffer.new_initialized(ctx, .{
        .size = @sizeOf(Vertex) * app_state.points_x_64 * 64,
        .usage = .{
            .vertex_buffer_bit = true,
        },
    }, Vertex{ .pos = .{ 0, 0, 0, 1 } }, pool);
    errdefer vertex_buffer.deinit(device);

    // apparently you don't need to create more than 1 depth buffer even if you have many
    // framebuffers
    const depth_image = blk: {
        const img = try device.createImage(&.{
            .image_type = .@"2d",
            .format = .d32_sfloat,
            .extent = .{
                .width = engine.window.extent.width,
                .height = engine.window.extent.height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{
                .depth_stencil_attachment_bit = true,
            },
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);
        errdefer device.destroyImage(img, null);

        const mem_reqs = device.getImageMemoryRequirements(img);
        const memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
        errdefer device.freeMemory(memory, null);
        try device.bindImageMemory(img, memory, 0);

        const view = try device.createImageView(&.{
            .image = img,
            .view_type = .@"2d",
            .format = .d32_sfloat,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .depth_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        errdefer device.destroyImageView(view, null);

        break :blk .{
            .image = img,
            .memory = memory,
            .view = view,
        };
    };
    errdefer {
        device.destroyImageView(depth_image.view, null);
        device.freeMemory(depth_image.memory, null);
        device.destroyImage(depth_image.image, null);
    }

    var voxels = try Buffer.new(ctx, .{ .size = @sizeOf(u32) * try std.math.powi(u32, app_state.voxels.side, 3) });
    errdefer voxels.deinit(device);
    var occlusion = try Buffer.new(ctx, .{ .size = @sizeOf(u32) * try std.math.powi(u32, app_state.voxels.side, 3) });
    errdefer occlusion.deinit(device);

    var screen = try Buffer.new(ctx, .{ .size = @sizeOf(f32) * 4 * 2 * engine.window.extent.width * engine.window.extent.height });
    errdefer screen.deinit(device);

    var screen_depth = try Buffer.new(ctx, .{ .size = @sizeOf(f32) * engine.window.extent.width * engine.window.extent.height });
    errdefer screen_depth.deinit(device);

    var reduction = try Buffer.new(ctx, .{ .size = @sizeOf(f32) * 4 * 3 + @sizeOf(f32) * 3 * app_state.reduction_points_x_64 * 64 });
    errdefer reduction.deinit(device);

    const frag_bindings = [_]vk.DescriptorSetLayoutBinding{
        uniforms.layout_binding(0),
        voxels.layout_binding(1),
        occlusion.layout_binding(2),
        screen.layout_binding(3),
        screen_depth.layout_binding(4),
        reduction.layout_binding(5),
    };
    const frag_desc_set_layout = try device.createDescriptorSetLayout(&.{
        .flags = .{},
        .binding_count = frag_bindings.len,
        .p_bindings = &frag_bindings,
    }, null);
    errdefer device.destroyDescriptorSetLayout(frag_desc_set_layout, null);

    const compute_bindings = [_]vk.DescriptorSetLayoutBinding{
        uniforms.layout_binding(0),
        vertex_buffer.layout_binding(1),
        voxels.layout_binding(2),
        occlusion.layout_binding(3),
        screen.layout_binding(4),
        screen_depth.layout_binding(5),
        reduction.layout_binding(6),
    };
    const compute_desc_set_layout = try device.createDescriptorSetLayout(&.{
        .flags = .{},
        .binding_count = compute_bindings.len,
        .p_bindings = &compute_bindings,
    }, null);
    errdefer device.destroyDescriptorSetLayout(compute_desc_set_layout, null);
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{
            .type = .uniform_buffer,
            .descriptor_count = 1,
        },
        .{
            .type = .storage_buffer,
            .descriptor_count = 6,
        },
    };
    const desc_pool = try device.createDescriptorPool(&.{
        .max_sets = 2,
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
    }, null);
    errdefer device.destroyDescriptorPool(desc_pool, null);
    var frag_desc_set: vk.DescriptorSet = undefined;
    var compute_desc_set: vk.DescriptorSet = undefined;
    try device.allocateDescriptorSets(&.{
        .descriptor_pool = desc_pool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&frag_desc_set_layout),
    }, @ptrCast(&frag_desc_set));
    try device.allocateDescriptorSets(&.{
        .descriptor_pool = desc_pool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&compute_desc_set_layout),
    }, @ptrCast(&compute_desc_set));
    const frag_desc_set_updates = [_]vk.WriteDescriptorSet{
        uniforms.write_desc_set(0, frag_desc_set),
        voxels.write_desc_set(1, frag_desc_set),
        occlusion.write_desc_set(2, frag_desc_set),
        screen.write_desc_set(3, frag_desc_set),
        screen_depth.write_desc_set(4, frag_desc_set),
        reduction.write_desc_set(5, frag_desc_set),
    };
    device.updateDescriptorSets(frag_desc_set_updates.len, &frag_desc_set_updates, 0, null);
    const compute_desc_set_updates = [_]vk.WriteDescriptorSet{
        uniforms.write_desc_set(0, compute_desc_set),
        vertex_buffer.write_desc_set(1, compute_desc_set),
        voxels.write_desc_set(2, compute_desc_set),
        occlusion.write_desc_set(3, compute_desc_set),
        screen.write_desc_set(4, compute_desc_set),
        screen_depth.write_desc_set(5, compute_desc_set),
        reduction.write_desc_set(6, compute_desc_set),
    };
    device.updateDescriptorSets(compute_desc_set_updates.len, &compute_desc_set_updates, 0, null);

    const compiler = utils.Glslc.Compiler{ .opt = .fast, .env = .vulkan1_3 };
    const vert_spv = blk: {
        const vert: utils.Glslc.Compiler.Code = .{ .path = .{
            .main = "./src/shader.glsl",
            .include = &[_][]const u8{},
            .definitions = &[_][]const u8{ "EYEFACE_RENDER", "EYEFACE_VERT" },
        } };
        // _ = compiler.dump_assembly(allocator, &vert);
        const res = try compiler.compile(
            allocator,
            &vert,
            .spirv,
            .vertex,
        );
        switch (res) {
            .Err => |msg| {
                std.debug.print("{s}\n", .{msg.msg});
                allocator.free(msg.msg);
                return msg.err;
            },
            .Ok => |ok| {
                errdefer allocator.free(ok);
                break :blk ok;
            },
        }
    };
    defer allocator.free(vert_spv);
    const frag_spv = blk: {
        const frag: utils.Glslc.Compiler.Code = .{ .path = .{
            .main = "./src/shader.glsl",
            .include = &[_][]const u8{},
            .definitions = &[_][]const u8{ "EYEFACE_RENDER", "EYEFACE_FRAG" },
        } };
        // _ = compiler.dump_assembly(allocator, &frag);
        const res = try compiler.compile(
            allocator,
            &frag,
            .spirv,
            .fragment,
        );
        switch (res) {
            .Err => |msg| {
                std.debug.print("{s}\n", .{msg.msg});
                allocator.free(msg.msg);
                return msg.err;
            },
            .Ok => |ok| {
                errdefer allocator.free(ok);
                break :blk ok;
            },
        }
    };
    defer allocator.free(frag_spv);

    const pass = blk: {
        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&vk.AttachmentReference{
                .attachment = 0,
                .layout = .color_attachment_optimal,
            }),
            .p_depth_stencil_attachment = @ptrCast(&vk.AttachmentReference{
                .attachment = 1,
                .layout = .depth_stencil_attachment_optimal,
            }),
        };

        const color_attachment = vk.AttachmentDescription{
            .format = swapchain.surface_format.format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .color_attachment_optimal,
        };

        const depth_attachment = vk.AttachmentDescription{
            .format = .d32_sfloat,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .depth_stencil_attachment_optimal,
        };

        const attachments = [_]vk.AttachmentDescription{ color_attachment, depth_attachment };

        const deps = [_]vk.SubpassDependency{
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{
                    .color_attachment_output_bit = true,
                },
                .dst_stage_mask = .{
                    .color_attachment_output_bit = true,
                },
                // src_access_mask: AccessFlags = .{},
                .dst_access_mask = .{
                    .color_attachment_write_bit = true,
                },
                // dependency_flags: DependencyFlags = .{},
            },
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{
                    .early_fragment_tests_bit = true,
                },
                .dst_stage_mask = .{
                    .early_fragment_tests_bit = true,
                },
                // src_access_mask: AccessFlags = .{},
                .dst_access_mask = .{
                    .depth_stencil_attachment_write_bit = true,
                },
                // dependency_flags: DependencyFlags = .{},
            },
        };

        break :blk try device.createRenderPass(&.{
            .attachment_count = @intCast(attachments.len),
            .p_attachments = &attachments,
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = @intCast(deps.len),
            .p_dependencies = &deps,
        }, null);
    };
    errdefer device.destroyRenderPass(pass, null);

    const compute_pipeline_layout = try device.createPipelineLayout(&.{
        // .flags: PipelineLayoutCreateFlags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&compute_desc_set_layout),
    }, null);
    errdefer device.destroyPipelineLayout(compute_pipeline_layout, null);
    const compute_pipelines = blk: {
        var pipelines = [_]struct {
            path: [:0]const u8,
            define: []const []const u8,
            group_x: u32 = 1,
            group_y: u32 = 1,
            group_z: u32 = 1,
            reduction_factor: ?u32 = null,
            pipeline: vk.Pipeline = undefined,
        }{
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_CLEAR_BUFS" },
                .group_x = @max(app_state.voxels.side * app_state.voxels.side * app_state.voxels.side / 64, blk1: {
                    const s = engine.window.extent.width * engine.window.extent.height;
                    break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
                }),
            },
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_ITERATE" },
                .group_x = app_state.points_x_64,
            },
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_REDUCE", "EYEFACE_REDUCE_MIN" },
                .reduction_factor = 256,
                .group_x = app_state.reduction_points_x_64,
            },
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_REDUCE", "EYEFACE_REDUCE_MAX" },
                .reduction_factor = 256,
                .group_x = app_state.reduction_points_x_64,
            },
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_PROJECT" },
                .group_x = app_state.points_x_64,
            },
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_OCCLUSION" },
                .group_x = app_state.voxels.side * app_state.voxels.side * app_state.voxels.side / 64,
            },
        };

        for (pipelines, 0..) |p, i| {
            const compute_spv = blk1: {
                const frag: utils.Glslc.Compiler.Code = .{ .path = .{
                    .main = p.path,
                    .include = &[_][]const u8{},
                    .definitions = p.define,
                } };
                // _ = compiler.dump_assembly(allocator, &frag);
                const res = try compiler.compile(
                    allocator,
                    &frag,
                    .spirv,
                    .compute,
                );
                switch (res) {
                    .Err => |msg| {
                        std.debug.print("{s}\n", .{msg.msg});
                        allocator.free(msg.msg);
                        return msg.err;
                    },
                    .Ok => |ok| {
                        errdefer allocator.free(ok);
                        break :blk1 ok;
                    },
                }
            };
            defer allocator.free(compute_spv);

            const compute = try device.createShaderModule(&.{
                .code_size = compute_spv.len * @sizeOf(u32),
                .p_code = @ptrCast(compute_spv.ptr),
            }, null);
            defer device.destroyShaderModule(compute, null);

            const cpci = vk.ComputePipelineCreateInfo{
                .stage = .{
                    .stage = .{
                        .compute_bit = true,
                    },
                    .module = compute,
                    .p_name = "main",
                },
                .layout = compute_pipeline_layout,
                .base_pipeline_index = undefined,
            };

            _ = try device.createComputePipelines(.null_handle, 1, @ptrCast(&cpci), null, @ptrCast(&pipelines[i].pipeline));
        }

        break :blk pipelines;
    };
    errdefer {
        for (compute_pipelines) |p| {
            device.destroyPipeline(p.pipeline, null);
        }
    }

    const pipeline_layout = try device.createPipelineLayout(&.{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&frag_desc_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    errdefer device.destroyPipelineLayout(pipeline_layout, null);

    const pipeline = blk: {
        const vert = try device.createShaderModule(&.{
            .code_size = vert_spv.len * @sizeOf(u32),
            .p_code = @ptrCast(vert_spv.ptr),
        }, null);
        defer device.destroyShaderModule(vert, null);

        const frag = try device.createShaderModule(&.{
            .code_size = frag_spv.len * @sizeOf(u32),
            .p_code = @ptrCast(frag_spv.ptr),
        }, null);
        defer device.destroyShaderModule(frag, null);

        const pssci = [_]vk.PipelineShaderStageCreateInfo{
            .{
                .stage = .{ .vertex_bit = true },
                .module = vert,
                .p_name = "main",
            },
            .{
                .stage = .{ .fragment_bit = true },
                .module = frag,
                .p_name = "main",
            },
        };

        const pvisci = vk.PipelineVertexInputStateCreateInfo{
            // .vertex_binding_description_count = 0,
            // .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
            // .vertex_attribute_description_count = Vertex.attribute_description.len,
            // .p_vertex_attribute_descriptions = &Vertex.attribute_description,
        };

        const piasci = vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };

        const pvsci = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
            .scissor_count = 1,
            .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
        };

        const prsci = vk.PipelineRasterizationStateCreateInfo{
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .cull_mode = .{ .back_bit = false },
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        };

        const pmsci = vk.PipelineMultisampleStateCreateInfo{
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = vk.FALSE,
            .min_sample_shading = 1,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const pcbas = vk.PipelineColorBlendAttachmentState{
            .blend_enable = vk.FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        };

        const pcbsci = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&pcbas),
            .blend_constants = [_]f32{ 0, 0, 0, 0 },
        };

        const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
        const pdsci = vk.PipelineDynamicStateCreateInfo{
            .flags = .{},
            .dynamic_state_count = dynstate.len,
            .p_dynamic_states = &dynstate,
        };

        const depth_stencil_info = vk.PipelineDepthStencilStateCreateInfo{
            .depth_test_enable = vk.TRUE,
            .depth_write_enable = vk.TRUE,
            .depth_compare_op = .less,
            .depth_bounds_test_enable = vk.FALSE,
            .stencil_test_enable = vk.FALSE,
            .front = .{
                .fail_op = .keep,
                .pass_op = .replace,
                .depth_fail_op = .keep,
                .compare_op = .always,
                .compare_mask = 0xFF,
                .write_mask = 0xFF,
                .reference = 1,
            },
            .back = .{
                .fail_op = .keep,
                .pass_op = .replace,
                .depth_fail_op = .keep,
                .compare_op = .always,
                .compare_mask = 0xFF,
                .write_mask = 0xFF,
                .reference = 1,
            },
            .min_depth_bounds = 0.0,
            .max_depth_bounds = 1.0,
        };

        const gpci = vk.GraphicsPipelineCreateInfo{
            .flags = .{},
            .stage_count = 2,
            .p_stages = &pssci,
            .p_vertex_input_state = &pvisci,
            .p_input_assembly_state = &piasci,
            .p_tessellation_state = null,
            .p_viewport_state = &pvsci,
            .p_rasterization_state = &prsci,
            .p_multisample_state = &pmsci,
            .p_depth_stencil_state = &depth_stencil_info,
            .p_color_blend_state = &pcbsci,
            .p_dynamic_state = &pdsci,
            .layout = pipeline_layout,
            .render_pass = pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = -1,
        };

        var pipeline: vk.Pipeline = undefined;
        _ = try device.createGraphicsPipelines(
            .null_handle,
            1,
            @ptrCast(&gpci),
            null,
            @ptrCast(&pipeline),
        );
        break :blk pipeline;
    };
    errdefer device.destroyPipeline(pipeline, null);

    const framebuffers = blk: {
        const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
        errdefer allocator.free(framebuffers);

        var i_2: usize = 0;
        errdefer for (framebuffers[0..i_2]) |fb| device.destroyFramebuffer(fb, null);
        for (framebuffers) |*fb| {
            const attachments = [_]vk.ImageView{ swapchain.swap_images[i_2].view, depth_image.view };
            fb.* = try device.createFramebuffer(&.{
                .render_pass = pass,
                .attachment_count = attachments.len,
                .p_attachments = &attachments,
                .width = swapchain.extent.width,
                .height = swapchain.extent.height,
                .layers = 1,
            }, null);
            i_2 += 1;
        }

        break :blk framebuffers;
    };
    errdefer {
        for (framebuffers) |fb| device.destroyFramebuffer(fb, null);
        allocator.free(framebuffers);
    }

    const command_buffers = blk: {
        const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
        errdefer allocator.free(cmdbufs);

        try device.allocateCommandBuffers(&.{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = @intCast(cmdbufs.len),
        }, cmdbufs.ptr);
        errdefer device.freeCommandBuffers(pool, @intCast(cmdbufs.len), cmdbufs.ptr);

        const clear = [_]vk.ClearValue{
            .{
                .color = .{
                    .float_32 = app_state.background_color.gamma_correct_inv().to_buf(),
                },
            },
            .{
                .depth_stencil = .{
                    .depth = 1.0,
                    .stencil = 0,
                },
            },
        };

        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(swapchain.extent.width),
            .height = @floatFromInt(swapchain.extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };
        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swapchain.extent,
        };

        for (cmdbufs, framebuffers) |cmdbuf, framebuffer| {
            try device.beginCommandBuffer(cmdbuf, &.{});

            for (compute_pipelines) |p| {
                device.cmdBindPipeline(cmdbuf, .compute, p.pipeline);
                device.cmdBindDescriptorSets(cmdbuf, .compute, compute_pipeline_layout, 0, 1, @ptrCast(&compute_desc_set), 0, null);

                var x = p.group_x;
                while (x >= 1) {
                    device.cmdDispatch(cmdbuf, x, p.group_y, p.group_z);
                    device.cmdPipelineBarrier(cmdbuf, .{
                        .compute_shader_bit = true,
                    }, .{
                        .compute_shader_bit = true,
                    }, .{}, 1, &[_]vk.MemoryBarrier{.{
                        .src_access_mask = .{
                            .shader_read_bit = true,
                            .shader_write_bit = true,
                        },
                        .dst_access_mask = .{
                            .shader_read_bit = true,
                            .shader_write_bit = true,
                        },
                    }}, 0, null, 0, null);

                    if (x == 1) {
                        break;
                    }

                    if (p.reduction_factor) |r| {
                        x = x / r + @as(u32, @intCast(@intFromBool(x % r > 0)));
                    } else {
                        break;
                    }
                }
            }

            // device.cmdPipelineBarrier(cmdbuf, .{
            //     .compute_shader_bit = true,
            // }, .{
            //     .vertex_input_bit = true,
            //     .vertex_shader_bit = true,
            // }, .{}, 1, &[_]vk.MemoryBarrier{.{
            //     .src_access_mask = .{
            //         .shader_write_bit = true,
            //     },
            //     .dst_access_mask = .{
            //         .shader_read_bit = true,
            //         .vertex_attribute_read_bit = true,
            //     },
            // }}, 1, (&[_]vk.BufferMemoryBarrier{.{
            //     .src_queue_family_index = ctx.graphics_queue.family,
            //     .dst_queue_family_index = ctx.graphics_queue.family,
            //     .buffer = vertex_buffer.buffer,
            //     .offset = 0,
            //     .size = vk.WHOLE_SIZE,
            //     .src_access_mask = .{
            //         .shader_write_bit = true,
            //     },
            //     .dst_access_mask = .{
            //         .vertex_attribute_read_bit = true,
            //         .shader_read_bit = true,
            //     },
            // }}), 0, null);

            device.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));
            device.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));

            device.cmdBeginRenderPass(cmdbuf, &.{
                .render_pass = pass,
                .framebuffer = framebuffer,
                .render_area = .{
                    .offset = .{ .x = 0, .y = 0 },
                    .extent = swapchain.extent,
                },
                .clear_value_count = clear.len,
                .p_clear_values = &clear,
            }, .@"inline");

            device.cmdBindPipeline(cmdbuf, .graphics, pipeline);
            device.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 0, 1, @ptrCast(&frag_desc_set), 0, null);
            device.cmdDraw(cmdbuf, 6, 1, 0, 0);

            device.cmdEndRenderPass(cmdbuf);
            try device.endCommandBuffer(cmdbuf);
        }

        break :blk cmdbufs;
    };
    errdefer {
        device.freeCommandBuffers(pool, @truncate(command_buffers.len), command_buffers.ptr);
        allocator.free(command_buffers);
    }

    return .{
        .swapchain = swapchain,
        .uniforms = uniforms,
        .vertex_buffer = vertex_buffer,
        .depth_image = depth_image.image,
        .depth_buffer_memory = depth_image.memory,
        .depth_image_view = depth_image.view,
        .voxel_buffer = voxels,
        .occlusion_buffer = occlusion,
        .screen_buffer = screen,
        .screen_depth_buffer = screen_depth,
        .reduction_buffer = reduction,
        .descriptor_pool = desc_pool,
        .frag_descriptor_set_layout = frag_desc_set_layout,
        .frag_descriptor_set = frag_desc_set,
        .compute_descriptor_set_layout = compute_desc_set_layout,
        .compute_descriptor_set = compute_desc_set,
        .pass = pass,
        .compute_pipeline_layout = compute_pipeline_layout,
        .compute_pipelines = blk: {
            const pipelines = try allocator.alloc(vk.Pipeline, compute_pipelines.len);
            for (compute_pipelines, 0..) |p, i| {
                pipelines[i] = p.pipeline;
            }
            break :blk pipelines;
        },
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,
        .framebuffers = framebuffers,
        .command_pool = pool,
        .command_buffers = command_buffers,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
    try self.swapchain.waitForAllFences(device);

    defer self.swapchain.deinit(device);

    defer self.uniforms.deinit(device);
    defer self.vertex_buffer.deinit(device);
    defer {
        device.destroyImageView(self.depth_image_view, null);
        device.freeMemory(self.depth_buffer_memory, null);
        device.destroyImage(self.depth_image, null);
    }
    defer {
        self.voxel_buffer.deinit(device);
        self.occlusion_buffer.deinit(device);
    }
    defer self.screen_buffer.deinit(device);

    defer self.screen_depth_buffer.deinit(device);
    defer self.reduction_buffer.deinit(device);
    defer device.destroyDescriptorSetLayout(self.frag_descriptor_set_layout, null);
    defer device.destroyDescriptorSetLayout(self.compute_descriptor_set_layout, null);
    defer device.destroyDescriptorPool(self.descriptor_pool, null);

    defer device.destroyRenderPass(self.pass, null);
    defer device.destroyPipelineLayout(self.compute_pipeline_layout, null);
    defer {
        for (self.compute_pipelines) |p| {
            device.destroyPipeline(p, null);
        }
        allocator.free(self.compute_pipelines);
    }
    defer device.destroyPipelineLayout(self.pipeline_layout, null);
    defer device.destroyPipeline(self.pipeline, null);

    defer {
        for (self.framebuffers) |fb| device.destroyFramebuffer(fb, null);
        allocator.free(self.framebuffers);
    }
    defer device.destroyCommandPool(self.command_pool, null);
    defer {
        device.freeCommandBuffers(self.command_pool, @truncate(self.command_buffers.len), self.command_buffers.ptr);
        allocator.free(self.command_buffers);
    }
}

pub fn present(
    self: *@This(),
    gui_renderer: *GuiEngine.GuiRenderer,
    ctx: *Engine.VulkanContext,
) !Swapchain.PresentState {
    const cmdbuf = self.command_buffers[self.swapchain.image_index];
    const gui_cmdbuf = gui_renderer.cmd_bufs[self.swapchain.image_index];

    return self.swapchain.present(&[_]vk.CommandBuffer{ cmdbuf, gui_cmdbuf }, ctx, &self.uniforms) catch |err| switch (err) {
        error.OutOfDateKHR => return .suboptimal,
        else => |narrow| return narrow,
    };
}
