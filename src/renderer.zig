const std = @import("std");

const vk = @import("vulkan");

const utils = @import("utils.zig");
const Fuse = utils.Fuse;

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Mat4x4 = math.Mat4x4;

const transform = @import("transform.zig");

const Engine = @import("engine.zig");

const gui = @import("gui.zig");
const GuiEngine = gui.GuiEngine;

const state = @import("state.zig");
const AppState = state.AppState;

const render_utils = @import("render_utils.zig");
const Swapchain = render_utils.Swapchain;
const UniformBuffer = render_utils.UniformBuffer;
const Buffer = render_utils.Buffer;
const Image = render_utils.Image;
const RenderPass = render_utils.RenderPass;
const GraphicsPipeline = render_utils.GraphicsPipeline;
const ComputePipeline = render_utils.ComputePipeline;
const DescriptorPool = render_utils.DescriptorPool;
const DescriptorSet = render_utils.DescriptorSet;
const Framebuffer = render_utils.Framebuffer;
const CmdBuffer = render_utils.CmdBuffer;

const main = @import("main.zig");
const allocator = main.allocator;

const Renderer = @This();

swapchain: Swapchain,
uniforms: UniformBuffer(Uniforms),
points_buffer: Buffer,
voxel_buffer: Buffer,
occlusion_buffer: Buffer,
g_buffer: Buffer,
screen_image: Image,
screen_depth_buffer: Buffer,
reduction_buffer: Buffer,
descriptor_pool: DescriptorPool,
compute_descriptor_set: DescriptorSet,
compute_pipelines: []ComputePipeline,
// framebuffers are objects containing views of swapchain images
command_pool: vk.CommandPool,
// command buffers are recordings of a bunch of commands that a gpu can execute
command_buffers: CmdBuffer,

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
    monitor_width: u32,
    monitor_height: u32,

    pub const TransformSet = transform.TransformSet(5);
};

pub fn init(engine: *Engine, app_state: *AppState) !@This() {
    var ctx = &engine.graphics;
    const device = &ctx.device;

    var swapchain = try Swapchain.init(ctx, engine.window.extent);
    errdefer swapchain.deinit(device);

    var uniforms = try UniformBuffer(Uniforms).new(app_state.uniforms(engine.window), ctx);
    errdefer uniforms.deinit(device);

    const cmd_pool = try device.createCommandPool(&.{
        .queue_family_index = ctx.graphics_queue.family,
        .flags = .{
            .reset_command_buffer_bit = true,
        },
    }, null);
    errdefer device.destroyCommandPool(cmd_pool, null);

    var points_buffer = try Buffer.new_initialized(ctx, .{
        .size = @sizeOf(Vertex) * app_state.max_points_x_64 * 64,
    }, Vertex{ .pos = .{ 0, 0, 0, 1 } }, cmd_pool);
    errdefer points_buffer.deinit(device);

    var voxels = try Buffer.new(ctx, .{ .size = @sizeOf(u32) * try std.math.powi(u32, app_state.voxels.max_side, 3) });
    errdefer voxels.deinit(device);
    var occlusion = try Buffer.new(ctx, .{ .size = @sizeOf(u32) * try std.math.powi(u32, app_state.voxels.max_side, 3) });
    errdefer occlusion.deinit(device);

    var g_buffer = try Buffer.new(ctx, .{ .size = @sizeOf(f32) * 4 * 2 * app_state.monitor_rez.width * app_state.monitor_rez.height });
    errdefer g_buffer.deinit(device);

    var screen = try Image.new(ctx, .{
        .img_type = .@"2d",
        .img_view_type = .@"2d",
        .format = .r16g16b16a16_sfloat,
        .extent = .{
            .width = app_state.monitor_rez.width,
            .height = app_state.monitor_rez.height,
            .depth = 1,
        },
        .usage = .{
            .transfer_src_bit = true,
            .storage_bit = true,
        },
        .view_aspect_mask = .{
            .color_bit = true,
        },
    });
    errdefer screen.deinit(device);
    var screen_depth = try Buffer.new(ctx, .{ .size = @sizeOf(f32) * app_state.monitor_rez.width * app_state.monitor_rez.height });
    errdefer screen_depth.deinit(device);

    var reduction = try Buffer.new(ctx, .{ .size = @sizeOf(f32) * 4 * 3 + @sizeOf(f32) * 3 * app_state.max_points_x_64 * 64 });
    errdefer reduction.deinit(device);

    var desc_pool = try DescriptorPool.new(device);
    errdefer desc_pool.deinit(device);

    var compute_set_builder = desc_pool.set_builder();
    defer compute_set_builder.deinit();
    try compute_set_builder.add(&uniforms);
    try compute_set_builder.add(&points_buffer);
    try compute_set_builder.add(&voxels);
    try compute_set_builder.add(&occlusion);
    try compute_set_builder.add(&g_buffer);
    try compute_set_builder.add(&screen);
    try compute_set_builder.add(&screen_depth);
    try compute_set_builder.add(&reduction);
    var compute_set = try compute_set_builder.build(device);
    errdefer compute_set.deinit(device);

    const compiler = utils.Glslc.Compiler{ .opt = .fast, .env = .vulkan1_3 };
    const compute_pipelines = blk: {
        const screen_sze = blk1: {
            const s = engine.window.extent.width * engine.window.extent.height;
            break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
        };
        const voxel_grid_sze = blk1: {
            const s = try std.math.powi(u32, app_state.voxels.side, 3);
            break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
        };
        var pipelines = [_]struct {
            path: [:0]const u8,
            define: []const []const u8,
            group_x: u32 = 1,
            group_y: u32 = 1,
            group_z: u32 = 1,
            reduction_factor: ?u32 = null,
            pipeline: ComputePipeline = undefined,
        }{
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_CLEAR_BUFS" },
                .group_x = @max(voxel_grid_sze, screen_sze),
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
                .group_x = voxel_grid_sze,
            },
            .{
                .path = "./src/shader.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_DRAW" },
                .group_x = screen_sze,
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

            pipelines[i].pipeline = try ComputePipeline.new(device, .{
                .shader = compute_spv,
                .desc_set_layouts = &[_]vk.DescriptorSetLayout{compute_set.layout},
            });
        }

        break :blk pipelines;
    };
    errdefer {
        for (compute_pipelines) |p| {
            p.pipeline.deinit(device);
        }
    }

    var cmdbuf = try CmdBuffer.init(device, .{ .pool = cmd_pool, .size = swapchain.swap_images.len });
    errdefer cmdbuf.deinit(device);

    try cmdbuf.begin(device);
    cmdbuf.transitionImg(device, .{
        .image = screen.image,
        .layout = .undefined,
        .new_layout = .general,
        .queue_family_index = ctx.graphics_queue.family,
    });
    for (compute_pipelines) |p| {
        cmdbuf.bindCompute(device, .{ .pipeline = p.pipeline, .desc_set = compute_set.set });
        var x = p.group_x;
        while (x >= 1) {
            cmdbuf.dispatch(device, .{ .x = x, .y = p.group_y, .z = p.group_z });
            cmdbuf.memBarrier(device, .{});

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
    cmdbuf.transitionImg(device, .{
        .image = screen.image,
        .layout = .undefined,
        .new_layout = .transfer_src_optimal,
        .queue_family_index = ctx.graphics_queue.family,
    });
    cmdbuf.transitionSwapchain(device, .{
        .layout = .undefined,
        .new_layout = .transfer_dst_optimal,
        .queue_family_index = ctx.graphics_queue.family,
        .swapchain = &swapchain,
    });
    cmdbuf.blitIntoSwapchain(device, .{
        .image = screen.image,
        .size = swapchain.extent,
        .swapchain = &swapchain,
    });
    cmdbuf.transitionSwapchain(device, .{
        .layout = .transfer_dst_optimal,
        .new_layout = .color_attachment_optimal,
        .queue_family_index = ctx.graphics_queue.family,
        .swapchain = &swapchain,
    });
    try cmdbuf.end(device);

    return .{
        .swapchain = swapchain,
        .uniforms = uniforms,
        .points_buffer = points_buffer,
        .voxel_buffer = voxels,
        .occlusion_buffer = occlusion,
        .g_buffer = g_buffer,
        .screen_image = screen,
        .screen_depth_buffer = screen_depth,
        .reduction_buffer = reduction,
        .descriptor_pool = desc_pool,
        .compute_descriptor_set = compute_set,
        .compute_pipelines = blk: {
            const pipelines = try allocator.alloc(ComputePipeline, compute_pipelines.len);
            for (compute_pipelines, 0..) |p, i| {
                pipelines[i] = p.pipeline;
            }
            break :blk pipelines;
        },
        .command_pool = cmd_pool,
        .command_buffers = cmdbuf,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
    try self.swapchain.waitForAllFences(device);

    defer self.swapchain.deinit(device);

    defer self.uniforms.deinit(device);
    defer self.points_buffer.deinit(device);
    defer {
        self.voxel_buffer.deinit(device);
        self.occlusion_buffer.deinit(device);
    }
    defer self.g_buffer.deinit(device);

    defer self.screen_image.deinit(device);
    defer self.screen_depth_buffer.deinit(device);
    defer self.reduction_buffer.deinit(device);
    defer self.compute_descriptor_set.deinit(device);
    defer self.descriptor_pool.deinit(device);

    defer {
        for (self.compute_pipelines) |p| {
            p.deinit(device);
        }
        allocator.free(self.compute_pipelines);
    }

    defer device.destroyCommandPool(self.command_pool, null);
    defer self.command_buffers.deinit(device);
}

pub fn present(
    self: *@This(),
    gui_renderer: *GuiEngine.GuiRenderer,
    ctx: *Engine.VulkanContext,
) !Swapchain.PresentState {
    const cmdbuf = self.command_buffers.bufs[self.swapchain.image_index];
    const gui_cmdbuf = gui_renderer.cmd_bufs[self.swapchain.image_index];

    return self.swapchain.present(&[_]vk.CommandBuffer{ cmdbuf, gui_cmdbuf }, ctx, &self.uniforms) catch |err| switch (err) {
        error.OutOfDateKHR => return .suboptimal,
        else => |narrow| return narrow,
    };
}
