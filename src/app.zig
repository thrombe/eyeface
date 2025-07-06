const std = @import("std");

const vk = @import("vulkan");

const utils_mod = @import("utils.zig");
const Fuse = utils_mod.Fuse;
const ShaderUtils = utils_mod.ShaderUtils;
const Telemetry = utils_mod.Tracy;
const cast = utils_mod.cast;

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Vec3 = math.Vec3;

const transform = @import("transform.zig");

const engine_mod = @import("engine.zig");
const Engine = engine_mod.Engine;
const c = engine_mod.c;
const Device = engine_mod.VulkanContext.Api.Device;

const gui = @import("gui.zig");
const GuiEngine = gui.GuiEngine;

const render_utils = @import("render_utils.zig");
const Swapchain = render_utils.Swapchain;
const Buffer = render_utils.Buffer;
const Image = render_utils.Image;
const ComputePipeline = render_utils.ComputePipeline;
const DescriptorPool = render_utils.DescriptorPool;
const DescriptorSet = render_utils.DescriptorSet;
const CmdBuffer = render_utils.CmdBuffer;

const world_mod = @import("world.zig");

const main = @import("main.zig");
const allocator = main.allocator;

pub const App = @This();

screen_image: Image,
resources: ResourceManager,
descriptor_pool: DescriptorPool,
command_pool: vk.CommandPool,

telemetry: Telemetry,

pub fn init(engine: *Engine) !@This() {
    var ctx = &engine.graphics;
    const device = &ctx.device;

    const res = try engine.window.get_res();

    var telemetry = try utils_mod.Tracy.init();
    errdefer telemetry.deinit();

    const cmd_pool = try device.createCommandPool(&.{
        .queue_family_index = ctx.graphics_queue.family,
        .flags = .{
            .reset_command_buffer_bit = true,
        },
    }, null);
    errdefer device.destroyCommandPool(cmd_pool, null);

    var screen = try Image.new(ctx, cmd_pool, .{
        .img_type = .@"2d",
        .img_view_type = .@"2d",
        .format = .r16g16b16a16_sfloat,
        .layout = .general,
        .extent = .{
            .width = res.width,
            .height = res.height,
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

    var resources = try ResourceManager.init(engine, cmd_pool, .{
        .monitor_rez = .{ .width = res.width, .height = res.height },
        // TODO: This should come from AppState, but AppState is not created yet.
        // This chicken-and-egg problem is a flaw in the current setup.
        // For now, hardcode or pass default.
        .max_points_x_64 = 1000000,
        .max_voxel_side = 500,
    });
    errdefer resources.deinit(device);

    var desc_pool = try DescriptorPool.new(device);
    errdefer desc_pool.deinit(device);

    return @This(){
        .screen_image = screen,
        .resources = resources,
        .descriptor_pool = desc_pool,
        .command_pool = cmd_pool,
        .telemetry = telemetry,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
    defer device.destroyCommandPool(self.command_pool, null);
    defer self.screen_image.deinit(device);
    defer self.resources.deinit(device);
    defer self.descriptor_pool.deinit(device);
    defer self.telemetry.deinit();
}

pub fn pre_reload(self: *@This()) !void {
    _ = self;
}

pub fn post_reload(self: *@This()) !void {
    _ = self;
}

pub fn tick(
    self: *@This(),
    engine: *Engine,
    app_state: *AppState,
    gui_renderer: *GuiEngine.GuiRenderer,
    gui_state: *GuiState,
    renderer_state: *RendererState,
) !bool {
    self.telemetry.mark_frame() catch |e| utils_mod.dump_error(e);
    self.telemetry.begin_sample(@src(), "frame.tick");
    defer self.telemetry.end_sample();
    self.telemetry.plot("last frame time (ms)", app_state.ticker.real.delta * std.time.ms_per_s);

    const ctx = &engine.graphics;

    if (engine.window.should_close()) return false;
    if (engine.window.is_minimized()) return true;

    gui_renderer.render_start();

    try app_state.tick(engine, self);

    {
        self.telemetry.begin_sample(@src(), "gui_state.tick");
        defer self.telemetry.end_sample();

        gui_state.tick(self, app_state);
    }
    {
        self.telemetry.begin_sample(@src(), "gui_renderer.render_end");
        defer self.telemetry.end_sample();

        try gui_renderer.render_end(&engine.graphics.device, &renderer_state.swapchain);
    }
    {
        self.telemetry.begin_sample(@src(), ".queue_wait_idle");
        defer self.telemetry.end_sample();
        try ctx.device.queueWaitIdle(ctx.graphics_queue.handle);
    }
    {
        self.telemetry.begin_sample(@src(), ".framerate_cap_sleep");
        defer self.telemetry.end_sample();

        const frametime = app_state.ticker.real.timer.read();
        const min_frametime_ns = std.time.ns_per_s / app_state.fps_cap;
        if (frametime < min_frametime_ns) {
            std.Thread.sleep(min_frametime_ns - frametime);
        }
    }
    {
        self.telemetry.begin_sample(@src(), ".gpu_buffer_uploads");
        defer self.telemetry.end_sample();
        try self.resources.upload(&ctx.device);
    }

    if (renderer_state.stages.update()) {
        _ = app_state.shader_fuse.fuse();
    }

    if (app_state.shader_fuse.unfuse() or app_state.reset_render_state.unfuse()) {
        self.telemetry.begin_sample(@src(), ".recreating_pipelines");
        defer self.telemetry.end_sample();
        try renderer_state.recreate_pipelines(engine, self, app_state);
    }

    if (app_state.cmdbuf_fuse.unfuse() or app_state.reset_render_state.unfuse()) {
        self.telemetry.begin_sample(@src(), ".recreating_command_buffers");
        defer self.telemetry.end_sample();
        try renderer_state.recreate_cmdbuf(engine, self, app_state);
    }

    {
        self.telemetry.begin_sample(@src(), ".present");
        defer self.telemetry.end_sample();
        try renderer_state.swapchain.present_start(ctx);
        const present_state = renderer_state.swapchain.present_end(
            &[_]vk.CommandBuffer{
                renderer_state.cmdbuffer.bufs[renderer_state.swapchain.image_index],
                gui_renderer.cmd_bufs[renderer_state.swapchain.image_index],
            },
            ctx,
        ) catch |err| switch (err) {
            error.OutOfDateKHR => blk: {
                _ = app_state.resize_fuse.fuse();
                break :blk .suboptimal;
            },
            else => |narrow| return narrow,
        };
        if (present_state == .suboptimal) {
            _ = app_state.resize_fuse.fuse();
        }
    }

    if (engine.window.resize_fuse.unfuse()) {
        _ = app_state.resize_fuse.fuse();
    }

    if (app_state.resize_fuse.unfuse()) {
        self.telemetry.begin_sample(@src(), ".recreating_swapchain");
        defer self.telemetry.end_sample();
        try ctx.device.queueWaitIdle(ctx.graphics_queue.handle);
        try renderer_state.recreate_swapchain(engine, app_state);
        gui_renderer.deinit(&engine.graphics.device);
        gui_renderer.* = try GuiEngine.GuiRenderer.init(engine, &renderer_state.swapchain);
    }

    return true;
}

pub const ResourceManager = struct {
    uniform: Uniforms,
    uniform_buf: Buffer,
    points_buffer: Buffer,
    voxel_buffer: Buffer,
    occlusion_buffer: Buffer,
    g_buffer: Buffer,
    screen_depth_buffer: Buffer,
    reduction_buffer: Buffer,

    pub fn init(engine: *Engine, pool: vk.CommandPool, v: struct {
        monitor_rez: struct { width: u32, height: u32 },
        max_points_x_64: u32,
        max_voxel_side: u32,
    }) !@This() {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        var uniform_buf = try Buffer.new(ctx, .{
            .size = @sizeOf(Uniforms.shader_type),
            .usage = .{ .uniform_buffer_bit = true },
            .memory_type = .{ .host_visible_bit = true, .host_coherent_bit = true },
            .desc_type = .uniform_buffer,
        });
        errdefer uniform_buf.deinit(device);

        var points_buffer = try Buffer.new_initialized(ctx, .{
            .size = v.max_points_x_64 * 64,
            .usage = .{ .storage_buffer_bit = true },
        }, [4]f32{ 0, 0, 0, 1 }, pool);
        errdefer points_buffer.deinit(device);

        var voxels = try Buffer.new(ctx, .{
            .size = @sizeOf(u32) * try std.math.powi(u32, v.max_voxel_side, 3),
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer voxels.deinit(device);

        var occlusion = try Buffer.new(ctx, .{
            .size = @sizeOf(u32) * try std.math.powi(u32, v.max_voxel_side, 3),
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer occlusion.deinit(device);

        var g_buffer = try Buffer.new(ctx, .{
            .size = @sizeOf(f32) * 4 * 2 * v.monitor_rez.width * v.monitor_rez.height,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer g_buffer.deinit(device);

        var screen_depth = try Buffer.new(ctx, .{
            .size = @sizeOf(f32) * v.monitor_rez.width * v.monitor_rez.height,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer screen_depth.deinit(device);

        var reduction = try Buffer.new(ctx, .{
            .size = @sizeOf(f32) * 4 * 3 + @sizeOf(f32) * 3 * v.max_points_x_64 * 64,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer reduction.deinit(device);

        return @This(){
            .uniform = std.mem.zeroes(Uniforms),
            .uniform_buf = uniform_buf,
            .points_buffer = points_buffer,
            .voxel_buffer = voxels,
            .occlusion_buffer = occlusion,
            .g_buffer = g_buffer,
            .screen_depth_buffer = screen_depth,
            .reduction_buffer = reduction,
        };
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        self.uniform_buf.deinit(device);
        self.points_buffer.deinit(device);
        self.voxel_buffer.deinit(device);
        self.occlusion_buffer.deinit(device);
        self.g_buffer.deinit(device);
        self.screen_depth_buffer.deinit(device);
        self.reduction_buffer.deinit(device);
    }

    pub fn add_binds(self: *@This(), builder: *render_utils.DescriptorSet.Builder, screen_image: *Image) !void {
        try builder.add(&self.uniform_buf, UniformBinds.uniforms.bind());
        try builder.add(&self.points_buffer, UniformBinds.points.bind());
        try builder.add(&self.voxel_buffer, UniformBinds.voxels.bind());
        try builder.add(&self.occlusion_buffer, UniformBinds.occlusion.bind());
        try builder.add(&self.g_buffer, UniformBinds.gbuffer.bind());
        try builder.add(screen_image, UniformBinds.screen.bind());
        try builder.add(&self.screen_depth_buffer, UniformBinds.screen_depth.bind());
        try builder.add(&self.reduction_buffer, UniformBinds.reduction.bind());
    }

    pub fn upload(self: *@This(), device: *Device) !void {
        const maybe_mapped = try device.mapMemory(self.uniform_buf.memory, 0, vk.WHOLE_SIZE, .{});
        const mapped = maybe_mapped orelse return error.MappingMemoryFailed;
        defer device.unmapMemory(self.uniform_buf.memory);

        const mem: *Uniforms.shader_type = @ptrCast(@alignCast(mapped));
        mem.* = ShaderUtils.shader_object(Uniforms.shader_type, self.uniform);
    }

    pub const DescSets = enum(u32) { compute };
    pub const UniformBinds = enum(u32) {
        uniforms,
        points,
        voxels,
        occlusion,
        gbuffer,
        screen,
        screen_depth,
        reduction,
        pub fn bind(self: @This()) u32 {
            return @intFromEnum(self);
        }
    };

    pub const PushConstants = struct {
        seed: i32,
    };

    pub const Uniforms = struct {
        transforms: [AppState.TransformSet.len]math.Mat4x4,
        camera: ShaderUtils.Camera3D,
        mouse: ShaderUtils.Mouse,
        frame: ShaderUtils.Frame,
        params: Params,

        const Params = struct {
            world_to_screen: math.Mat4x4,
            occlusion_color: Vec4,
            sparse_color: Vec4,
            background_color: Vec4,
            voxel_grid_center: Vec4,
            voxel_grid_side: i32,
            voxel_grid_compensation_perc: f32,
            occlusion_multiplier: f32,
            occlusion_attenuation: f32,
            depth_range: f32,
            depth_offset: f32,
            depth_attenuation: f32,
            points: i32,
            iterations: i32,
            voxelization_points: i32,
            voxelization_iterations: i32,
            reduction_points: i32,
            lambda: f32,
            visual_scale: f32,
            visual_transform_lambda: f32,
        };

        pub const shader_type = ShaderUtils.shader_type(@This());
    };
};

pub const RendererState = struct {
    swapchain: Swapchain,
    cmdbuffer: CmdBuffer,
    compute_desc_set: DescriptorSet,
    stages: ShaderStageManager,
    pipelines: Pipelines,

    // not owned
    pool: vk.CommandPool,

    const Pipelines = struct {
        clear_bufs: ComputePipeline,
        iterate: ComputePipeline,
        reduce_min: ComputePipeline,
        reduce_max: ComputePipeline,
        project: ComputePipeline,
        occlusion: ComputePipeline,
        draw: ComputePipeline,

        fn deinit(self: *@This(), device: *Device) void {
            self.clear_bufs.deinit(device);
            self.iterate.deinit(device);
            self.reduce_min.deinit(device);
            self.reduce_max.deinit(device);
            self.project.deinit(device);
            self.occlusion.deinit(device);
            self.draw.deinit(device);
        }
    };

    pub fn init(app: *App, engine: *Engine, app_state: *AppState) !@This() {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        var arena = std.heap.ArenaAllocator.init(allocator.*);
        defer arena.deinit();
        const alloc = arena.allocator();

        var gen = try utils_mod.ShaderUtils.GlslBindingGenerator.init();
        defer gen.deinit();
        try gen.add_struct("Mouse", ShaderUtils.Mouse);
        try gen.add_struct("Camera3D", ShaderUtils.Camera3D);
        try gen.add_struct("Frame", ShaderUtils.Frame);
        try gen.add_struct("Params", ResourceManager.Uniforms.Params);
        try gen.add_struct("Uniforms", ResourceManager.Uniforms);
        try gen.add_struct("PushConstants", ResourceManager.PushConstants);
        try gen.add_enum("_set", ResourceManager.DescSets);
        try gen.add_enum("_bind", ResourceManager.UniformBinds);
        try gen.dump_shader("src/uniforms.glsl");

        const includes = try alloc.dupe([]const u8, &[_][]const u8{"src"});
        var stages_info = std.ArrayList(utils_mod.ShaderCompiler.ShaderInfo).init(alloc);
        try stages_info.appendSlice(&[_]utils_mod.ShaderCompiler.ShaderInfo{
            .{
                .name = "clear_bufs",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "CLEAR_BUFS_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
            .{
                .name = "iterate",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "ITERATE_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
            .{
                .name = "reduce_min",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "REDUCE_PASS", "REDUCE_MIN_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
            .{
                .name = "reduce_max",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "REDUCE_PASS", "REDUCE_MAX_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
            .{
                .name = "project",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "PROJECT_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
            .{
                .name = "occlusion",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "OCCLUSION_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
            .{
                .name = "draw",
                .stage = .compute,
                .path = "src/shader.glsl",
                .define = try alloc.dupe([]const u8, &[_][]const u8{ "DRAW_PASS", "COMPUTE_PASS" }),
                .include = includes,
            },
        });

        var stages = try ShaderStageManager.init(stages_info.items);
        errdefer stages.deinit();

        var swapchain = try Swapchain.init(ctx, engine.window.extent, .{});
        errdefer swapchain.deinit(device);

        var self: @This() = .{
            .stages = stages,
            .pipelines = undefined,
            .compute_desc_set = undefined,
            .swapchain = swapchain,
            .pool = app.command_pool,
            .cmdbuffer = undefined,
        };

        try self.create_pipelines(engine, app, false);
        errdefer self.compute_desc_set.deinit(device);
        errdefer self.pipelines.deinit(device);

        self.cmdbuffer = try self.create_cmdbuf(engine, app, app_state);
        errdefer self.cmdbuffer.deinit(device);

        return self;
    }

    pub fn recreate_pipelines(self: *@This(), engine: *Engine, app: *App, app_state: *AppState) !void {
        try self.create_pipelines(engine, app, true);
        _ = app_state.cmdbuf_fuse.fuse();
    }

    pub fn recreate_swapchain(self: *@This(), engine: *Engine, app_state: *AppState) !void {
        try self.swapchain.recreate(&engine.graphics, engine.window.extent, .{});
        _ = app_state.cmdbuf_fuse.fuse();
    }

    pub fn recreate_cmdbuf(self: *@This(), engine: *Engine, app: *App, app_state: *AppState) !void {
        const device = &engine.graphics.device;
        const cmdbuffer = try self.create_cmdbuf(engine, app, app_state);
        self.cmdbuffer.deinit(device);
        self.cmdbuffer = cmdbuffer;
    }

    fn create_pipelines(self: *@This(), engine: *Engine, app: *App, initialized: bool) !void {
        const device = &engine.graphics.device;

        var set_builder = app.descriptor_pool.set_builder();
        defer set_builder.deinit();
        try app.resources.add_binds(&set_builder, &app.screen_image);

        var compute_set = try set_builder.build(device);
        errdefer compute_set.deinit(device);

        const desc_set_layouts = &[_]vk.DescriptorSetLayout{compute_set.layout};
        const push_constant_ranges = &[_]vk.PushConstantRange{.{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(ResourceManager.PushConstants),
        }};

        if (initialized) self.pipelines.deinit(device);

        self.pipelines.clear_bufs = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("clear_bufs").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });
        self.pipelines.iterate = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("iterate").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });
        self.pipelines.reduce_min = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("reduce_min").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });
        self.pipelines.reduce_max = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("reduce_max").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });
        self.pipelines.project = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("project").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });
        self.pipelines.occlusion = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("occlusion").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });
        self.pipelines.draw = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("draw").?.code,
            .desc_set_layouts = desc_set_layouts,
            .push_constant_ranges = push_constant_ranges,
        });

        if (initialized) self.compute_desc_set.deinit(device);
        self.compute_desc_set = compute_set;
    }

    pub fn create_cmdbuf(self: *@This(), engine: *Engine, app: *App, app_state: *AppState) !CmdBuffer {
        const ctx = &engine.graphics;
        const device = &ctx.device;
        const alloc = app_state.arena.allocator();

        var cmdbuf = try CmdBuffer.init(device, .{ .pool = app.command_pool, .size = self.swapchain.swap_images.len });
        errdefer cmdbuf.deinit(device);

        const screen_sze = blk1: {
            const s = engine.window.extent.width * engine.window.extent.height;
            break :blk1 s / 64 + @as(u32, @intFromBool(s % 64 > 0));
        };
        const voxel_grid_sze = blk1: {
            const s = try std.math.powi(u32, @intCast(app_state.voxels.side), 3);
            break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
        };

        const passes = [_]struct {
            pipeline: ComputePipeline,
            group_x: u32,
            reduction_factor: ?u32,
        }{
            .{ .pipeline = self.pipelines.clear_bufs, .group_x = @max(voxel_grid_sze, screen_sze), .reduction_factor = null },
            .{ .pipeline = self.pipelines.iterate, .group_x = @intCast(app_state.points_x_64), .reduction_factor = null },
            .{ .pipeline = self.pipelines.reduce_min, .group_x = @intCast(app_state.reduction_points_x_64), .reduction_factor = 256 },
            .{ .pipeline = self.pipelines.reduce_max, .group_x = @intCast(app_state.reduction_points_x_64), .reduction_factor = 256 },
            .{ .pipeline = self.pipelines.project, .group_x = @intCast(app_state.points_x_64), .reduction_factor = null },
            .{ .pipeline = self.pipelines.occlusion, .group_x = voxel_grid_sze, .reduction_factor = null },
            .{ .pipeline = self.pipelines.draw, .group_x = screen_sze, .reduction_factor = null },
        };

        try cmdbuf.begin(device);
        for (passes) |p| {
            cmdbuf.bindCompute(device, .{ .pipeline = p.pipeline, .desc_sets = &.{self.compute_desc_set.set} });

            var x = p.group_x;
            while (x >= 1) {
                const constants = try alloc.create(ResourceManager.PushConstants);
                constants.* = .{ .seed = app_state.rng.random().int(i32) };
                cmdbuf.push_constants(device, p.pipeline.layout, std.mem.asBytes(constants), .{ .compute_bit = true });
                cmdbuf.dispatch(device, .{ .x = x, .y = 1, .z = 1 });
                cmdbuf.memBarrier(device, .{});

                if (x == 1) break;
                if (p.reduction_factor) |r| {
                    x = x / r + @as(u32, @intCast(@intFromBool(x % r > 0)));
                } else {
                    break;
                }
            }
        }
        cmdbuf.draw_into_swapchain(device, .{
            .image = app.screen_image.image,
            .image_layout = .general,
            .size = self.swapchain.extent,
            .swapchain = &self.swapchain,
            .queue_family = ctx.graphics_queue.family,
        });
        try cmdbuf.end(device);

        return cmdbuf;
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        try self.swapchain.waitForAll(device);
        defer self.swapchain.deinit(device);
        defer self.cmdbuffer.deinit(device);
        defer self.compute_desc_set.deinit(device);
        defer self.stages.deinit();
        defer self.pipelines.deinit(device);
    }
};

const ShaderStageManager = struct {
    shaders: utils_mod.ShaderCompiler.Stages,
    compiler: utils_mod.ShaderCompiler.Compiler,

    pub fn init(stages: []const utils_mod.ShaderCompiler.ShaderInfo) !@This() {
        var comp = try utils_mod.ShaderCompiler.Compiler.init(.{ .opt = .fast, .env = .vulkan1_3 }, stages);
        errdefer comp.deinit();

        return .{
            .shaders = try utils_mod.ShaderCompiler.Stages.init(&comp, stages),
            .compiler = comp,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.shaders.deinit();
        self.compiler.deinit();
    }

    pub fn update(self: *@This()) bool {
        return self.shaders.update(&self.compiler);
    }
};

pub const AppState = struct {
    pub const TransformSet = transform.TransformSet(5);

    ticker: utils_mod.SimulationTicker,

    monitor_rez: struct { width: u32, height: u32 },
    mouse: extern struct { x: i32 = 0, y: i32 = 0, left: bool = false, right: bool = false } = .{},
    pos: Vec3,
    camera: math.Camera,
    controller: world_mod.Components.Controller = .{},

    frame: u32 = 0,
    fps_cap: u32 = 500,
    focus: bool = false,

    transform_generator: TransformSet.Builder.Generator = .{},
    transforms: TransformSet.Builder,
    target_transforms: TransformSet.Builder,

    t: f32 = 0,
    lambda: f32 = 1.0,
    visual_scale: f32 = 4.0,
    visual_transform_lambda: f32 = 1.0,
    pause_t: bool = false,
    pause_generator: bool = false,
    points_x_64: i32 = 50000,
    max_points_x_64: u32 = 1000000,
    iterations: i32 = 20,
    voxel_grid_compensation_perc: f32 = 0.1,
    voxelization_points_x_64: i32 = 50000,
    voxelization_iterations: i32 = 4,
    reduction_points_x_64: i32 = 50000,

    background_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#271212ff"),
    occlusion_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#3a0e0eff"),
    sparse_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#fbc8ccff"),
    occlusion_multiplier: f32 = 1.16,
    occlusion_attenuation: f32 = 0.55,
    depth_range: f32 = 5.35,
    depth_offset: f32 = 2.5,
    depth_attenuation: f32 = 1.156,

    voxels: struct {
        // world space coords of center of the the voxel grid
        center: Vec4 = .{},
        // number of voxels along 1 edge (side ** 3 is the entire volume)
        side: i32 = 300,
        max_side: u32 = 500,
    } = .{},

    rng: std.Random.Xoshiro256,
    reset_render_state: Fuse = .{},
    resize_fuse: Fuse = .{},
    cmdbuf_fuse: Fuse = .{},
    shader_fuse: Fuse = .{},
    arena: std.heap.ArenaAllocator,

    pub fn init(window: *engine_mod.Window, app: *App) !@This() {
        _ = app;
        const mouse = window.poll_mouse();
        const sze = try window.get_res();

        var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
        const generator = TransformSet.Builder.Generator{};

        return .{
            .ticker = try .init(),
            .monitor_rez = .{ .width = sze.width, .height = sze.height },
            .camera = math.Camera.init(math.Camera.constants.basis.vulkan, math.Camera.constants.basis.opengl),
            .pos = .{ .z = -5 },
            .controller = .{},
            .mouse = .{ .x = mouse.x, .y = mouse.y, .left = mouse.left },
            .transforms = transform.sirpinski_pyramid(),
            .target_transforms = generator.generate(rng.random()),
            .transform_generator = generator,
            .rng = rng,
            .arena = std.heap.ArenaAllocator.init(allocator.*),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.arena.deinit();
    }

    pub fn pre_reload(self: *@This()) !void {
        _ = self;
    }

    pub fn post_reload(self: *@This()) !void {
        _ = self.resize_fuse.fuse();
        _ = self.shader_fuse.fuse();
        _ = self.cmdbuf_fuse.fuse();
    }

    pub fn tick(self: *@This(), engine: *Engine, app: *App) !void {
        app.telemetry.begin_sample(@src(), "app_state.tick");
        defer app.telemetry.end_sample();
        defer _ = self.arena.reset(.retain_capacity);

        self.ticker.tick_real();
        engine.window.tick();

        try self.tick_local_input(engine, app);

        if (!self.pause_t) {
            // - [Lerp smoothing is broken](https://youtu.be/LSNQuFEDOyQ?si=-_bGNwqZFC_j5dJF&t=3012)
            const e = 1.0 - std.math.exp(-self.lambda * self.ticker.real.delta);
            self.transforms = self.transforms.mix(&self.target_transforms, e);
            self.t = std.math.lerp(self.t, 1.0, e);
        }
        if (1.0 - self.t < 0.01 and !self.pause_generator) {
            self.t = 0;
            self.target_transforms = self.transform_generator.generate(self.rng.random());
        }

        try self.tick_prepare_render(engine, app);
    }

    fn tick_local_input(self: *@This(), engine: *Engine, app: *App) !void {
        const window = engine.window;
        const delta = self.ticker.real.delta;

        var input = window.input();

        // local input tick
        {
            app.telemetry.begin_sample(@src(), ".local_input");
            defer app.telemetry.end_sample();

            var mouse = &input.mouse;
            var kb = &input.keys;

            const imgui_io = &c.ImGui_GetIO()[0];
            if (imgui_io.WantCaptureMouse) {
                mouse.left = .none;
                mouse.right = .none;
            }
            if (imgui_io.WantCaptureKeyboard) {}

            // if (kb.p.just_pressed()) {
            //     try render_utils.dump_image_to_file(
            //         &app.screen_image,
            //         &engine.graphics,
            //         app.command_pool,
            //         window.extent,
            //         "images",
            //     );
            // }

            if (mouse.left.just_pressed() and !self.focus) {
                self.focus = true;
                imgui_io.ConfigFlags |= c.ImGuiConfigFlags_NoMouse;
                window.hide_cursor(true);
            }
            if (kb.escape.just_pressed() and !self.focus) {
                window.queue_close();
            }
            if (kb.escape.just_pressed() and self.focus) {
                self.focus = false;
                imgui_io.ConfigFlags &= ~c.ImGuiConfigFlags_NoMouse;
                window.hide_cursor(false);
            }

            self.mouse.left = mouse.left.pressed();
            self.mouse.x = @intFromFloat(mouse.x);
            self.mouse.y = @intFromFloat(mouse.y);

            self.frame += 1;

            if (!self.focus) {
                mouse.dx = 0;
                mouse.dy = 0;
            }
        }

        // Camera Logic
        {
            const mouse = &input.mouse;
            const kb = &input.keys;

            const rot = self.camera.rot_quat(self.controller.pitch, self.controller.yaw);
            const fwd = rot.rotate_vector(self.camera.world_basis.fwd);
            const right = rot.rotate_vector(self.camera.world_basis.right);

            var speed = self.controller.speed;
            if (kb.shift.pressed()) {
                speed *= 2.0;
            }
            if (kb.ctrl.pressed()) {
                speed *= 0.1;
            }

            if (kb.w.pressed()) {
                self.pos = self.pos.add(fwd.scale(delta * speed));
            }
            if (kb.a.pressed()) {
                self.pos = self.pos.sub(right.scale(delta * speed));
            }
            if (kb.s.pressed()) {
                self.pos = self.pos.sub(fwd.scale(delta * speed));
            }
            if (kb.d.pressed()) {
                self.pos = self.pos.add(right.scale(delta * speed));
            }

            self.controller.did_move = kb.w.pressed() or kb.a.pressed() or kb.s.pressed() or kb.d.pressed();
            self.controller.did_rotate = @abs(mouse.dx) + @abs(mouse.dy) > 0.0001;

            if (self.controller.did_rotate) {
                self.controller.yaw += @as(f32, @floatCast(mouse.dx)) * self.controller.sensitivity_scale * self.controller.sensitivity;
                self.controller.pitch += @as(f32, @floatCast(mouse.dy)) * self.controller.sensitivity_scale * self.controller.sensitivity;
                self.controller.pitch = std.math.clamp(self.controller.pitch, -std.math.pi / 2.0 + 0.001, std.math.pi / 2.0 - 0.001);
            }
        }
    }

    fn tick_prepare_render(self: *@This(), engine: *Engine, app: *App) !void {
        try self.prepare_uniforms(&app.resources, engine.window);
    }

    pub fn prepare_uniforms(self: *@This(), resources: *ResourceManager, window: *engine_mod.Window) !void {
        const rot = self.camera.rot_quat(self.controller.pitch, self.controller.yaw);
        const fwd = rot.rotate_vector(self.camera.world_basis.fwd);
        const right = rot.rotate_vector(self.camera.world_basis.right);
        const up = rot.rotate_vector(self.camera.world_basis.up);

        resources.uniform = .{
            .transforms = self.transforms.build().transforms,
            .camera = .{
                .eye = self.pos,
                .fwd = fwd,
                .right = right,
                .up = up,
                .meta = .{
                    .did_move = @intFromBool(self.controller.did_move),
                    .did_rotate = @intFromBool(self.controller.did_rotate),
                    .did_change = @intFromBool(self.controller.did_move or self.controller.did_rotate),
                },
            },
            .mouse = .{
                .x = self.mouse.x,
                .y = self.mouse.y,
                .left = @intFromBool(self.mouse.left),
                .right = @intFromBool(self.mouse.right),
            },
            .frame = .{
                .frame = self.frame,
                .time = self.ticker.real.time_f,
                .deltatime = self.ticker.real.delta,
                .width = @intCast(window.extent.width),
                .height = @intCast(window.extent.height),
                .monitor_width = @intCast(self.monitor_rez.width),
                .monitor_height = @intCast(self.monitor_rez.height),
            },
            .params = .{
                .world_to_screen = self.camera.world_to_screen_mat(.{
                    .width = self.monitor_rez.width,
                    .height = self.monitor_rez.height,
                    .pos = self.pos,
                    .pitch = self.controller.pitch,
                    .yaw = self.controller.yaw,
                    .far = 100.0,
                }),
                .occlusion_color = self.occlusion_color,
                .sparse_color = self.sparse_color,
                .background_color = self.background_color,
                .voxel_grid_center = self.voxels.center,
                .voxel_grid_side = self.voxels.side,
                .voxel_grid_compensation_perc = self.voxel_grid_compensation_perc,
                .occlusion_multiplier = self.occlusion_multiplier,
                .occlusion_attenuation = self.occlusion_attenuation,
                .depth_range = self.depth_range,
                .depth_offset = self.depth_offset,
                .depth_attenuation = self.depth_attenuation,
                .points = self.points_x_64 * 64,
                .iterations = self.iterations,
                .voxelization_points = self.voxelization_points_x_64 * 64,
                .voxelization_iterations = self.voxelization_iterations,
                .reduction_points = self.reduction_points_x_64 * 64,
                .lambda = self.lambda,
                .visual_scale = self.visual_scale,
                .visual_transform_lambda = self.visual_transform_lambda,
            },
        };
    }
};

pub const GuiState = struct {
    const TransformSet = AppState.TransformSet;
    const Constraints = TransformSet.Builder.Generator.Constraints;
    const ShearConstraints = TransformSet.Builder.Generator.ShearConstraints;
    const Vec3Constraints = TransformSet.Builder.Generator.Vec3Constraints;

    frame_times: [60]f32 = std.mem.zeroes([60]f32),
    frame_times_i: usize = 0,
    total: f32 = 0,

    pub fn tick(self: *@This(), app: *App, state: *AppState) void {
        _ = app;
        const delta = state.ticker.real.delta;
        const generator = &state.transform_generator;

        self.frame_times_i = @rem(self.frame_times_i + 1, self.frame_times.len);
        self.total -= self.frame_times[self.frame_times_i];
        self.frame_times[self.frame_times_i] = delta * std.time.ms_per_s;
        self.total += self.frame_times[self.frame_times_i];
        const frametime = self.total / @as(f32, @floatFromInt(self.frame_times.len));

        c.ImGui_SetNextWindowPos(.{ .x = 5, .y = 5 }, c.ImGuiCond_Once);
        defer c.ImGui_End();
        if (c.ImGui_Begin("SIKE", null, c.ImGuiWindowFlags_None)) {
            c.ImGui_Text("Application average %.3f ms/frame (%.1f FPS)", frametime, std.time.ms_per_s / frametime);

            c.ImGui_Text("State");
            self.editState(state);

            c.ImGui_Text("Scale");
            self.editVec3Constraints("Scale", &generator.scale);

            c.ImGui_Text("Rotation");
            self.editVec3Constraints("Rotation", &generator.rot);

            c.ImGui_Text("Translation");
            self.editVec3Constraints("Translation", &generator.translate);

            c.ImGui_Text("Shear");
            self.editShear("Shear", &generator.shear);
        }
    }

    fn editState(self: *@This(), state: *AppState) void {
        _ = self;

        var reset: bool = false;

        _ = c.ImGui_SliderFloat("Speed", &state.controller.speed, 0.1, 10.0);
        _ = c.ImGui_SliderFloat("Sensitivity", &state.controller.sensitivity, 0.001, 2.0);
        _ = c.ImGui_SliderInt("FPS cap", @ptrCast(&state.fps_cap), 5, 500);

        _ = c.ImGui_SliderFloat("Visual Scale", &state.visual_scale, 0.01, 10.0);
        _ = c.ImGui_SliderFloat("Visual Transform Lambda", &state.visual_transform_lambda, 0.0, 25.0);
        _ = c.ImGui_SliderFloat("voxel grid compensation perc", &state.voxel_grid_compensation_perc, -1.0, 1.0);
        reset = c.ImGui_SliderInt("points (x 64)", @ptrCast(&state.points_x_64), 0, @intCast(state.max_points_x_64)) or reset;
        state.voxelization_points_x_64 = @min(state.voxelization_points_x_64, state.points_x_64);
        _ = c.ImGui_SliderInt("voxelization points (x 64)", @ptrCast(&state.voxelization_points_x_64), 0, @intCast(state.points_x_64));
        state.reduction_points_x_64 = @min(state.reduction_points_x_64, state.points_x_64);
        reset = c.ImGui_SliderInt("reduction points (x 64)", @ptrCast(&state.reduction_points_x_64), 0, @intCast(state.points_x_64)) or reset;
        _ = c.ImGui_SliderInt("iterations", @ptrCast(&state.iterations), 0, 100);
        _ = c.ImGui_SliderInt("voxelization iterations", @ptrCast(&state.voxelization_iterations), 0, 20);

        _ = c.ImGui_ColorEdit3("Background Color", @ptrCast(&state.background_color), c.ImGuiColorEditFlags_Float);
        _ = c.ImGui_ColorEdit3("Occlusion Color", @ptrCast(&state.occlusion_color), c.ImGuiColorEditFlags_Float);
        _ = c.ImGui_ColorEdit3("Sparse Color", @ptrCast(&state.sparse_color), c.ImGuiColorEditFlags_Float);

        _ = c.ImGui_SliderFloat("Occlusion Multiplier", &state.occlusion_multiplier, 0.01, 4.0);
        _ = c.ImGui_SliderFloat("Occlusion Attenuation", &state.occlusion_attenuation, 0.1, 4.0);
        _ = c.ImGui_SliderFloat("Depth Range", &state.depth_range, 0.1, 20.0);
        _ = c.ImGui_SliderFloat("Depth Offset", &state.depth_offset, 0.1, 10.0);
        _ = c.ImGui_SliderFloat("Depth Attenuation", &state.depth_attenuation, 0.01, 2.0);

        _ = c.ImGui_SliderFloat("Lambda", &state.lambda, 0.1, 25.0);
        _ = c.ImGui_Checkbox("Pause t (pause_t)", &state.pause_t);
        _ = c.ImGui_Checkbox("Pause Generator (pause_generator)", &state.pause_generator);

        _ = c.ImGui_DragFloat3("Voxels Center", @ptrCast(&state.voxels.center));

        reset = c.ImGui_SliderInt("Voxel Side", @ptrCast(&state.voxels.side), 1, @intCast(state.voxels.max_side)) or reset;

        if (reset) {
            _ = state.cmdbuf_fuse.fuse();
        }
    }

    fn editVec3Constraints(self: *@This(), comptime label: [:0]const u8, constraints: *Vec3Constraints) void {
        c.ImGui_PushID(label);
        defer c.ImGui_PopID();
        self.editConstraint(".X", &constraints.x);
        self.editConstraint(".Y", &constraints.y);
        self.editConstraint(".Z", &constraints.z);
    }

    fn editConstraint(self: *@This(), comptime label: [:0]const u8, constraint: *Constraints) void {
        _ = self;

        const width = 75.0;

        c.ImGui_SetNextItemWidth(width);
        _ = c.ImGui_DragFloat(label ++ " Min", &constraint.min);
        c.ImGui_SameLine();
        c.ImGui_SetNextItemWidth(width);
        _ = c.ImGui_DragFloat(label ++ " Max", &constraint.max);
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox(label ++ " Flip Sign", &constraint.flip_sign);
    }

    fn editShear(self: *@This(), comptime label: [:0]const u8, shear: *ShearConstraints) void {
        c.ImGui_PushID(label);
        defer c.ImGui_PopID();
        self.editConstraint(".X.y", &shear.x.y);
        self.editConstraint(".X.z", &shear.x.z);
        self.editConstraint(".Y.x", &shear.y.x);
        self.editConstraint(".Y.z", &shear.y.z);
        self.editConstraint(".Z.x", &shear.z.x);
        self.editConstraint(".Z.y", &shear.z.y);
    }
};
