const std = @import("std");

const vk = @import("vulkan");

const utils = @import("utils.zig");
const Fuse = utils.Fuse;

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Mat4x4 = math.Mat4x4;

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

uniforms: UniformBuffer(Uniforms),
voxels: Buffer,
g_buffer: Buffer,
screen_image: Image,
descriptor_pool: DescriptorPool,
compute_descriptor_set: DescriptorSet,
command_pool: vk.CommandPool,
stages: ShaderStageManager,

const Device = Engine.VulkanContext.Api.Device;

pub const Uniforms = extern struct {
    camera: Camera,
    mouse: Mouse,

    frame: u32,
    time: f32,
    deltatime: f32,
    width: u32,
    height: u32,
    monitor_width: u32,
    monitor_height: u32,

    voxel_grid_side: u32,
    voxel_debug_view: u32,
    pad1: u32 = 0,
    pad2: u32 = 0,
    pad3: u32 = 0,

    background_color1: Vec4,
    background_color2: Vec4,

    light_dir: Vec4,
    fractal_iterations: u32,
    march_iterations: u32,
    gather_iterations: u32,
    gather_step_factor: f32,
    march_step_factor: f32,
    escape_r: f32,
    fractal_scale: f32,
    fractal_density: f32,
    gi_samples: u32,
    temporal_blend_factor: f32,
    t_max: f32,
    dt_min: f32,

    pub const Mouse = extern struct { x: i32, y: i32, left: u32, right: u32 };
    pub const Camera = extern struct {
        eye: Vec4,
        fwd: Vec4,
        right: Vec4,
        up: Vec4,
        meta: CameraMeta,

        pub const CameraMeta = extern struct {
            did_change: u32 = 0,
            did_move: u32 = 0,
            did_rotate: u32 = 0,
            pad: u32 = 0,
        };
    };
};

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

    var uniforms = try UniformBuffer(Uniforms).new(app_state.uniforms(engine.window), ctx);
    errdefer uniforms.deinit(device);

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

    var voxels = try Buffer.new(ctx, .{
        .size = @sizeOf(f32) * 4 * try std.math.powi(u32, app_state.voxels.max_side, 3),
    });
    errdefer voxels.deinit(device);

    var g_buffer = try Buffer.new(ctx, .{
        .size = @sizeOf(f32) * 4 * app_state.monitor_rez.width * app_state.monitor_rez.height,
    });
    errdefer g_buffer.deinit(device);

    var desc_pool = try DescriptorPool.new(device);
    errdefer desc_pool.deinit(device);

    var compute_set_builder = desc_pool.set_builder();
    defer compute_set_builder.deinit();
    try compute_set_builder.add(&uniforms);
    try compute_set_builder.add(&voxels);
    try compute_set_builder.add(&g_buffer);
    try compute_set_builder.add(&screen);
    var compute_set = try compute_set_builder.build(device);
    errdefer compute_set.deinit(device);

    const stages = try ShaderStageManager.init();
    errdefer stages.deinit();

    return .{
        .uniforms = uniforms,
        .voxels = voxels,
        .g_buffer = g_buffer,
        .screen_image = screen,
        .descriptor_pool = desc_pool,
        .compute_descriptor_set = compute_set,
        .command_pool = cmd_pool,
        .stages = stages,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
    defer device.destroyCommandPool(self.command_pool, null);
    defer self.uniforms.deinit(device);
    defer self.voxels.deinit(device);
    defer self.g_buffer.deinit(device);
    defer self.screen_image.deinit(device);
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
                const s = try std.math.powi(u32, app_state.voxels.side, 3);
                break :blk1 s / 64 + @as(u32, @intCast(@intFromBool(s % 64 > 0)));
            };
            var pipelines = [_]struct {
                typ: ShaderStageManager.ShaderStage,
                group_x: u32 = 1,
                group_y: u32 = 1,
                group_z: u32 = 1,
                pipeline: ComputePipeline = undefined,
            }{
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

        var swapchain = try Swapchain.init(ctx, engine.window.extent);
        errdefer swapchain.deinit(device);

        var cmdbuf = try CmdBuffer.init(device, .{ .pool = app.command_pool, .size = swapchain.swap_images.len });
        errdefer cmdbuf.deinit(device);

        try cmdbuf.begin(device);
        for (compute_pipelines) |p| {
            cmdbuf.bindCompute(device, .{ .pipeline = p.pipeline, .desc_set = app.compute_descriptor_set.set });
            cmdbuf.dispatch(device, .{ .x = p.group_x, .y = p.group_y, .z = p.group_z });
            cmdbuf.memBarrier(device, .{});
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
        draw,
    };
    const CompilerUtils = render_utils.ShaderCompiler(struct {
        pub fn get_metadata(_: CompilerUtils.ShaderInfo) !@This() {
            return .{};
        }
    }, ShaderStage);

    pub fn init() !@This() {
        var comp = try CompilerUtils.Compiler.init(&[_]CompilerUtils.ShaderInfo{
            .{
                .typ = .draw,
                .stage = .compute,
                .path = "./src/mandlebulb.glsl",
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
    monitor_rez: struct { width: u32, height: u32 },
    mouse: extern struct { x: i32 = 0, y: i32 = 0, left: bool = false, right: bool = false } = .{},
    camera: math.Camera,
    camera_meta: Uniforms.Camera.CameraMeta = .{},

    frame: u32 = 0,
    time: f32 = 0,
    deltatime: f32 = 0,

    voxels: struct {
        side: u32 = 100,
        max_side: u32 = 500,
    } = .{},

    background_color1: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#ff9999ff"),
    background_color2: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#33334cff"),

    light_dir: Vec4 = .{ .x = 1.0, .y = 1.0, .z = 1.0 },
    fractal_iterations: u32 = 10,
    march_iterations: u32 = 512,
    gather_iterations: u32 = 1024,
    gather_step_factor: f32 = 0.6,
    march_step_factor: f32 = 2.0,
    escape_r: f32 = 1.2,
    fractal_scale: f32 = 0.45,
    fractal_density: f32 = 128.0,
    gi_samples: u32 = 64,
    temporal_blend_factor: f32 = 0.9,
    t_max: f32 = 50.0,
    dt_min: f32 = 0.0001,
    voxel_debug_view: bool = false,

    rng: std.Random.Xoshiro256,

    pub fn init(window: *Engine.Window) !@This() {
        const mouse = window.poll_mouse();
        const sze = try window.get_res();

        const rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

        return .{
            .monitor_rez = .{ .width = sze.width, .height = sze.height },
            .camera = math.Camera.init(Vec4{ .z = 2 }, math.Camera.constants.basis.opengl),
            .mouse = .{ .x = mouse.x, .y = mouse.y, .left = mouse.left },
            .rng = rng,
        };
    }

    pub fn tick(self: *@This(), lap: u64, window: *Engine.Window) void {
        const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));
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
        self.camera_meta.did_move = @intCast(@intFromBool(w or a or s or d));
        self.camera_meta.did_rotate = @intCast(@intFromBool((dx | dy) != 0));
        self.camera_meta.did_change = @intCast(@intFromBool((self.camera_meta.did_move | self.camera_meta.did_rotate) > 0));
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
    }

    pub fn uniforms(self: *const @This(), window: *Engine.Window) Uniforms {
        const rot = self.camera.rot_quat();

        const fwd = rot.rotate_vector(self.camera.basis.fwd);
        const right = rot.rotate_vector(self.camera.basis.right);
        const up = rot.rotate_vector(self.camera.basis.up);
        const eye = self.camera.pos;

        return .{
            .camera = .{
                .eye = eye,
                .fwd = fwd,
                .right = right,
                .up = up,
                .meta = self.camera_meta,
            },
            .mouse = .{
                .x = self.mouse.x,
                .y = self.mouse.y,
                .left = @intCast(@intFromBool(self.mouse.left)),
                .right = @intCast(@intFromBool(self.mouse.right)),
            },
            .background_color1 = self.background_color1,
            .background_color2 = self.background_color2,
            .frame = self.frame,
            .time = self.time,
            .deltatime = self.deltatime,
            .width = window.extent.width,
            .height = window.extent.height,
            .monitor_width = self.monitor_rez.width,
            .monitor_height = self.monitor_rez.height,
            .voxel_grid_side = self.voxels.side,
            .light_dir = self.light_dir,
            .fractal_iterations = self.fractal_iterations,
            .march_iterations = self.march_iterations,
            .gather_iterations = self.gather_iterations,
            .gather_step_factor = self.gather_step_factor,
            .march_step_factor = self.march_step_factor,
            .escape_r = self.escape_r,
            .fractal_scale = self.fractal_scale,
            .fractal_density = self.fractal_density,
            .gi_samples = self.gi_samples,
            .temporal_blend_factor = self.temporal_blend_factor,
            .voxel_debug_view = @intCast(@intFromBool(self.voxel_debug_view)),
            .t_max = self.t_max,
            .dt_min = self.dt_min,
        };
    }
};

pub const GuiState = struct {
    frame_times: [10]f32 = std.mem.zeroes([10]f32),
    frame_times_i: usize = 10,

    pub fn tick(self: *@This(), state: *AppState, lap: u64) void {
        const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));

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
        }
    }

    fn editState(self: *@This(), state: *AppState) void {
        _ = self;

        _ = c.ImGui_SliderFloat("Speed", &state.camera.speed, 0.1, 10.0);
        _ = c.ImGui_SliderFloat("Sensitivity", &state.camera.sensitivity, 0.001, 2.0);

        if (c.ImGui_Button("Reset time")) {
            state.time = 0.0;
            state.deltatime = 0.0;
            state.frame = 0;
        }

        _ = c.ImGui_ColorEdit3("Background Color 1", @ptrCast(&state.background_color1), c.ImGuiColorEditFlags_Float);
        _ = c.ImGui_ColorEdit3("Background Color 2", @ptrCast(&state.background_color2), c.ImGuiColorEditFlags_Float);

        _ = c.ImGui_Checkbox("voxel debug view", @ptrCast(&state.voxel_debug_view));

        _ = c.ImGui_SliderFloat3("light dir", @ptrCast(&state.light_dir), 0.0, 1.0);
        state.light_dir = state.light_dir.normalize3D();
        _ = c.ImGui_SliderInt("Fractal iterations", @ptrCast(&state.fractal_iterations), 0, 50);
        _ = c.ImGui_SliderInt("March iterations", @ptrCast(&state.march_iterations), 0, 1024);
        _ = c.ImGui_SliderInt("Gather iterations", @ptrCast(&state.gather_iterations), 0, 4096);
        _ = c.ImGui_SliderFloat("gather step factor", &state.gather_step_factor, 0.01, 2.0);
        _ = c.ImGui_SliderFloat("march step factor", &state.march_step_factor, 0.01, 16.0);
        _ = c.ImGui_SliderFloat("escape radius", &state.escape_r, 0.01, 16.0);
        _ = c.ImGui_SliderFloat("fractal scale", &state.fractal_scale, 0.01, 2.0);
        _ = c.ImGui_SliderFloat("fractal density", &state.fractal_density, 0.01, 512.0);
        _ = c.ImGui_SliderInt("GI samples", @ptrCast(&state.gi_samples), 0, 4096);
        _ = c.ImGui_SliderFloat("temporal blend factor", &state.temporal_blend_factor, 0.01, 1.0);
        _ = c.ImGui_SliderFloat("t max", &state.t_max, 0.1, 1000.0);
        var pow = @log10(state.dt_min);
        _ = c.ImGui_SliderFloat("dt min (10^this)", &pow, -10.0, 0.0);
        state.dt_min = std.math.pow(f32, 10.0, pow);
    }
};
