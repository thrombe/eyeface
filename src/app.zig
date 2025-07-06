const std = @import("std");

const vk = @import("vulkan");

const utils = @import("utils.zig");
const Fuse = utils.Fuse;
const ShaderUtils = utils.ShaderUtils;

const math = @import("math.zig");
const Vec4 = math.Vec4;

const transform = @import("transform.zig");

const Engine = @import("engine.zig");
const c = Engine.c;

const gui = @import("gui.zig");
const GuiEngine = gui.GuiEngine;

const render_utils = @import("render_utils.zig");
const Swapchain = render_utils.Swapchain;
const UniformBuffer = render_utils.UniformBuffer;
const Buffer = render_utils.Buffer;
const Image = render_utils.Image;
const ComputePipeline = render_utils.ComputePipeline;
const DescriptorPool = render_utils.DescriptorPool;
const DescriptorSet = render_utils.DescriptorSet;
const CmdBuffer = render_utils.CmdBuffer;

const main = @import("main.zig");
const allocator = main.allocator;

pub const App = @This();

uniforms: UniformBuffer,
points_buffer: Buffer,
voxel_buffer: Buffer,
occlusion_buffer: Buffer,
g_buffer: Buffer,
screen_image: Image,
screen_depth_buffer: Buffer,
reduction_buffer: Buffer,
descriptor_pool: DescriptorPool,
compute_descriptor_set: DescriptorSet,
// framebuffers are objects containing views of swapchain images
command_pool: vk.CommandPool,
stages: ShaderStageManager,

const Device = Engine.VulkanContext.Api.Device;

pub fn init(engine: *Engine, app_state: *AppState) !@This() {
    var ctx = &engine.graphics;
    const device = &ctx.device;

    const cmd_pool = try device.createCommandPool(&.{
        .queue_family_index = ctx.graphics_queue.family,
        .flags = .{
            .reset_command_buffer_bit = true,
        },
    }, null);
    errdefer device.destroyCommandPool(cmd_pool, null);

    var uniforms = try UniformBuffer.new(try app_state.uniforms(engine.window), ctx);
    errdefer uniforms.deinit(device);

    var points_buffer = try Buffer.new_initialized(ctx, .{
        .size = @sizeOf(f32) * 4 * app_state.max_points_x_64 * 64,
    }, [4]f32{ 0, 0, 0, 1 }, cmd_pool);
    errdefer points_buffer.deinit(device);

    var voxels = try Buffer.new(ctx, .{
        .size = @sizeOf(u32) * try std.math.powi(u32, app_state.voxels.max_side, 3),
    });
    errdefer voxels.deinit(device);
    var occlusion = try Buffer.new(ctx, .{
        .size = @sizeOf(u32) * try std.math.powi(u32, app_state.voxels.max_side, 3),
    });
    errdefer occlusion.deinit(device);

    var g_buffer = try Buffer.new(ctx, .{
        .size = @sizeOf(f32) * 4 * 2 * app_state.monitor_rez.width * app_state.monitor_rez.height,
    });
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
    try screen.transition(ctx, cmd_pool, .undefined, .general);

    var screen_depth = try Buffer.new(ctx, .{
        .size = @sizeOf(f32) * app_state.monitor_rez.width * app_state.monitor_rez.height,
    });
    errdefer screen_depth.deinit(device);

    var reduction = try Buffer.new(ctx, .{
        .size = @sizeOf(f32) * 4 * 3 + @sizeOf(f32) * 3 * app_state.max_points_x_64 * 64,
    });
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

    const stages = try ShaderStageManager.init();
    errdefer stages.deinit();

    return .{
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
        .command_pool = cmd_pool,
        .stages = stages,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
    defer device.destroyCommandPool(self.command_pool, null);
    defer self.uniforms.deinit(device);
    defer self.points_buffer.deinit(device);
    defer self.voxel_buffer.deinit(device);
    defer self.occlusion_buffer.deinit(device);
    defer self.g_buffer.deinit(device);
    defer self.screen_image.deinit(device);
    defer self.screen_depth_buffer.deinit(device);
    defer self.reduction_buffer.deinit(device);
    defer self.compute_descriptor_set.deinit(device);
    defer self.descriptor_pool.deinit(device);
    defer self.stages.deinit();
}

pub fn present(
    self: *@This(),
    dynamic_state: *RendererState,
    gui_renderer: *GuiEngine.GuiRenderer,
    ctx: *Engine.VulkanContext,
) !Swapchain.PresentState {
    const cmdbuf = dynamic_state.cmdbuffer.bufs[dynamic_state.swapchain.image_index];
    const gui_cmdbuf = gui_renderer.cmd_bufs[dynamic_state.swapchain.image_index];

    return dynamic_state.swapchain.present(&[_]vk.CommandBuffer{ cmdbuf, gui_cmdbuf }, ctx, &self.uniforms) catch |err| switch (err) {
        error.OutOfDateKHR => return .suboptimal,
        else => |narrow| return narrow,
    };
}

pub const RendererState = struct {
    swapchain: Swapchain,
    cmdbuffer: CmdBuffer,
    compute_pipelines: []ComputePipeline,

    // not owned
    pool: vk.CommandPool,

    pub fn init(app: *App, engine: *Engine, app_state: *AppState) !@This() {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        const compute_pipelines = blk: {
            const screen_sze = blk1: {
                const s = engine.window.extent.width * engine.window.extent.height;
                break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
            };
            const voxel_grid_sze = blk1: {
                const s = try std.math.powi(u32, @intCast(app_state.voxels.side), 3);
                break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
            };
            var pipelines = [_]struct {
                typ: ShaderStageManager.ShaderStage,
                group_x: u32 = 1,
                group_y: u32 = 1,
                group_z: u32 = 1,
                reduction_factor: ?u32 = null,
                pipeline: ComputePipeline = undefined,
            }{
                .{
                    .typ = .clear_bufs,
                    .group_x = @max(voxel_grid_sze, screen_sze),
                },
                .{
                    .typ = .iterate,
                    .group_x = @intCast(app_state.points_x_64),
                },
                .{
                    .typ = .reduce_min,
                    .reduction_factor = 256,
                    .group_x = @intCast(app_state.reduction_points_x_64),
                },
                .{
                    .typ = .reduce_max,
                    .reduction_factor = 256,
                    .group_x = @intCast(app_state.reduction_points_x_64),
                },
                .{
                    .typ = .project,
                    .group_x = @intCast(app_state.points_x_64),
                },
                .{
                    .typ = .occlusion,
                    .group_x = voxel_grid_sze,
                },
                .{
                    .typ = .draw,
                    .group_x = screen_sze,
                },
            };

            for (pipelines, 0..) |p, i| {
                pipelines[i].pipeline = try ComputePipeline.new(device, .{
                    .shader = app.stages.shaders.map.get(p.typ).code,
                    .desc_set_layouts = &[_]vk.DescriptorSetLayout{app.compute_descriptor_set.layout},
                });
            }

            break :blk pipelines;
        };
        errdefer {
            for (compute_pipelines) |p| {
                p.pipeline.deinit(device);
            }
        }

        var swapchain = try Swapchain.init(ctx, engine.window.extent, .{});
        errdefer swapchain.deinit(device);

        var cmdbuf = try CmdBuffer.init(device, .{ .pool = app.command_pool, .size = swapchain.swap_images.len });
        errdefer cmdbuf.deinit(device);

        try cmdbuf.begin(device);
        for (compute_pipelines) |p| {
            cmdbuf.bindCompute(device, .{ .pipeline = p.pipeline, .desc_set = app.compute_descriptor_set.set });
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
        cmdbuf.drawIntoSwapchain(device, .{
            .image = app.screen_image.image,
            .image_layout = .general,
            .size = swapchain.extent,
            .swapchain = &swapchain,
            .queue_family = ctx.graphics_queue.family,
        });
        try cmdbuf.end(device);

        return .{
            .compute_pipelines = blk: {
                const pipelines = try allocator.alloc(ComputePipeline, compute_pipelines.len);
                for (compute_pipelines, 0..) |p, i| {
                    pipelines[i] = p.pipeline;
                }
                break :blk pipelines;
            },
            .swapchain = swapchain,
            .cmdbuffer = cmdbuf,
            .pool = app.command_pool,
        };
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        try self.swapchain.waitForAllFences(device);

        defer self.swapchain.deinit(device);
        defer self.cmdbuffer.deinit(device);

        defer {
            for (self.compute_pipelines) |p| {
                p.deinit(device);
            }
            allocator.free(self.compute_pipelines);
        }
    }
};

const ShaderStageManager = struct {
    shaders: CompilerUtils.Stages,
    compiler: CompilerUtils.Compiler,

    const ShaderStage = enum {
        clear_bufs,
        iterate,
        reduce_min,
        reduce_max,
        project,
        occlusion,
        draw,
    };
    const CompilerUtils = utils.ShaderCompiler(struct {
        pub fn get_metadata(_: CompilerUtils.ShaderInfo) !@This() {
            return .{};
        }
    }, ShaderStage);

    pub fn init() !@This() {
        var comp = try CompilerUtils.Compiler.init(.{ .opt = .fast, .env = .vulkan1_3 }, &[_]CompilerUtils.ShaderInfo{
            .{
                .typ = .clear_bufs,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_CLEAR_BUFS" },
                .include = &[_][]const u8{"./src"},
            },
            .{
                .typ = .iterate,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_ITERATE" },
                .include = &[_][]const u8{"./src"},
            },
            .{
                .typ = .reduce_min,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_REDUCE", "EYEFACE_REDUCE_MIN" },
                .include = &[_][]const u8{"./src"},
            },
            .{
                .typ = .reduce_max,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_REDUCE", "EYEFACE_REDUCE_MAX" },
                .include = &[_][]const u8{"./src"},
            },
            .{
                .typ = .project,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_PROJECT" },
                .include = &[_][]const u8{"./src"},
            },
            .{
                .typ = .occlusion,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_OCCLUSION" },
                .include = &[_][]const u8{"./src"},
            },
            .{
                .typ = .draw,
                .stage = .compute,
                .path = "./src/eyeface.glsl",
                .define = &[_][]const u8{ "EYEFACE_COMPUTE", "EYEFACE_DRAW" },
                .include = &[_][]const u8{"./src"},
            },
        });
        errdefer comp.deinit();

        return .{
            .shaders = try CompilerUtils.Stages.init(&comp),
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

    monitor_rez: struct { width: u32, height: u32 },
    mouse: extern struct { x: i32 = 0, y: i32 = 0, left: bool = false, right: bool = false } = .{},
    camera: math.Camera,

    frame: u32 = 0,
    time: f32 = 0,
    deltatime: f32 = 0,
    fps_cap: u32 = 500,

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
    uniform_buffer: []u8,
    uniform_shader_dumped: bool = false,

    pub fn init(window: *Engine.Window) !@This() {
        const mouse = window.poll_mouse();
        const sze = try window.get_res();

        var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

        const generator = TransformSet.Builder.Generator{};

        return .{
            .monitor_rez = .{ .width = sze.width, .height = sze.height },
            .camera = math.Camera.init(Vec4{ .z = -5 }, math.Camera.constants.basis.vulkan),
            .mouse = .{ .x = mouse.x, .y = mouse.y, .left = mouse.left },
            .transforms = transform.sirpinski_pyramid(),
            .target_transforms = generator.generate(rng.random()),
            .transform_generator = generator,
            .rng = rng,
            .uniform_buffer = try allocator.alloc(u8, 0),
        };
    }

    pub fn deinit(self: *@This()) void {
        allocator.free(self.uniform_buffer);
    }

    pub fn tick(self: *@This(), lap: u64, engine: *Engine, app: *App) !void {
        const window = engine.window;
        const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        // std.debug.print("fps: {d}\n", .{@as(u32, @intFromFloat(1.0 / delta))});
        const w = window.is_pressed(c.GLFW_KEY_W);
        const a = window.is_pressed(c.GLFW_KEY_A);
        const s = window.is_pressed(c.GLFW_KEY_S);
        const d = window.is_pressed(c.GLFW_KEY_D);
        const shift = window.is_pressed(c.GLFW_KEY_LEFT_SHIFT);
        const ctrl = window.is_pressed(c.GLFW_KEY_LEFT_CONTROL);
        const mouse = window.poll_mouse();

        var dx: i32 = 0;
        var dy: i32 = 0;
        if (mouse.left) {
            dx = mouse.x - self.mouse.x;
            dy = mouse.y - self.mouse.y;
        }
        self.camera.tick(delta, .{ .dx = dx, .dy = dy }, .{
            .w = w,
            .a = a,
            .s = s,
            .d = d,
            .shift = shift,
            .ctrl = ctrl,
        });

        self.mouse.left = mouse.left;
        self.mouse.x = mouse.x;
        self.mouse.y = mouse.y;

        self.frame += 1;
        self.time += delta;
        self.deltatime = delta;

        if (!self.pause_t) {
            // - [Lerp smoothing is broken](https://youtu.be/LSNQuFEDOyQ?si=-_bGNwqZFC_j5dJF&t=3012)
            const e = 1.0 - std.math.exp(-self.lambda * delta);
            self.transforms = self.transforms.mix(&self.target_transforms, e);
            self.t = std.math.lerp(self.t, 1.0, e);
        }
        if (1.0 - self.t < 0.01 and !self.pause_generator) {
            self.t = 0;
            self.target_transforms = self.transform_generator.generate(self.rng.random());
        }

        const p = window.is_pressed(c.GLFW_KEY_P);
        if (p) {
            try render_utils.dump_image_to_file(
                &app.screen_image,
                &engine.graphics,
                app.command_pool,
                window.extent,
                "./images",
            );
        }
    }

    pub fn uniforms(self: *@This(), window: *Engine.Window) ![]u8 {
        const transforms = self.transforms.build().transforms;

        const uniform = .{
            .transforms = transforms,
            .world_to_screen = self.camera.world_to_screen_mat(window.extent.width, window.extent.height),
            .eye = self.camera.pos,
            .mouse = ShaderUtils.Mouse{
                .x = self.mouse.x,
                .y = self.mouse.y,
                .left = @intCast(@intFromBool(self.mouse.left)),
                .right = @intCast(@intFromBool(self.mouse.right)),
            },
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
            .frame = self.frame,
            .time = self.time,
            .deltatime = self.deltatime,
            .lambda = self.lambda,
            .visual_scale = self.visual_scale,
            .visual_transform_lambda = self.visual_transform_lambda,
            .width = window.extent.width,
            .height = window.extent.height,
            .monitor_width = self.monitor_rez.width,
            .monitor_height = self.monitor_rez.height,
        };

        const ubo = ShaderUtils.create_uniform_object(@TypeOf(uniform), uniform);
        const ubo_buffer = std.mem.asBytes(&ubo);

        if (self.uniform_buffer.len != ubo_buffer.len) {
            allocator.free(self.uniform_buffer);
            self.uniform_buffer = try allocator.alloc(u8, ubo_buffer.len);
        }

        if (!self.uniform_shader_dumped) {
            self.uniform_shader_dumped = true;

            try ShaderUtils.dump_glsl_uniform(ubo, "./src/eyeface_uniforms.glsl");
        }

        @memcpy(self.uniform_buffer, ubo_buffer);

        return self.uniform_buffer;
    }
};

pub const GuiState = struct {
    const TransformSet = AppState.TransformSet;
    const Constraints = TransformSet.Builder.Generator.Constraints;
    const ShearConstraints = TransformSet.Builder.Generator.ShearConstraints;
    const Vec3Constraints = TransformSet.Builder.Generator.Vec3Constraints;

    frame_times: [10]f32 = std.mem.zeroes([10]f32),
    frame_times_i: usize = 10,

    pub fn tick(self: *@This(), state: *AppState, lap: u64) void {
        const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        const generator = &state.transform_generator;

        self.frame_times_i += 1;
        self.frame_times_i = @rem(self.frame_times_i, self.frame_times.len);
        self.frame_times[self.frame_times_i] = delta * std.time.ms_per_s;
        const frametime = std.mem.max(f32, &self.frame_times);

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

        _ = c.ImGui_SliderFloat("Speed", &state.camera.speed, 0.1, 10.0);
        _ = c.ImGui_SliderFloat("Sensitivity", &state.camera.sensitivity, 0.001, 2.0);
        _ = c.ImGui_SliderInt("FPS cap", @ptrCast(&state.fps_cap), 5, 500);

        _ = c.ImGui_SliderFloat("Visual Scale", &state.visual_scale, 0.01, 10.0);
        _ = c.ImGui_SliderFloat("Visual Transform Lambda", &state.visual_transform_lambda, 0.0, 25.0);
        _ = c.ImGui_SliderFloat("voxel grid compensation perc", &state.voxel_grid_compensation_perc, -1.0, 1.0);
        _ = c.ImGui_SliderInt("points (x 64)", @ptrCast(&state.points_x_64), 0, @intCast(state.max_points_x_64));
        state.voxelization_points_x_64 = @min(state.voxelization_points_x_64, state.points_x_64);
        _ = c.ImGui_SliderInt("voxelization points (x 64)", @ptrCast(&state.voxelization_points_x_64), 0, @intCast(state.points_x_64));
        state.reduction_points_x_64 = @min(state.reduction_points_x_64, state.points_x_64);
        _ = c.ImGui_SliderInt("reduction points (x 64)", @ptrCast(&state.reduction_points_x_64), 0, @intCast(state.points_x_64));
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

        _ = c.ImGui_SliderInt("Voxel Side", @ptrCast(&state.voxels.side), 1, @intCast(state.voxels.max_side));
    }

    fn editVec3Constraints(self: *@This(), comptime label: [:0]const u8, constraints: *Vec3Constraints) void {
        c.ImGui_PushID(label);
        self.editConstraint(".X", &constraints.x);
        self.editConstraint(".Y", &constraints.y);
        self.editConstraint(".Z", &constraints.z);
        c.ImGui_PopID();
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
        self.editConstraint(".X.y", &shear.x.y);
        self.editConstraint(".X.z", &shear.x.z);
        self.editConstraint(".Y.x", &shear.y.x);
        self.editConstraint(".Y.z", &shear.y.z);
        self.editConstraint(".Z.x", &shear.z.x);
        self.editConstraint(".Z.y", &shear.z.y);
        c.ImGui_PopID();
    }
};
