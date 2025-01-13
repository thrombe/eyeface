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
    world_to_screen: Mat4x4,
    eye: Vec4,
    fwd: Vec4,
    right: Vec4,
    up: Vec4,
    mouse: extern struct { x: i32, y: i32, left: u32, right: u32 },
    background_color: Vec4,
    frame: u32,
    time: f32,
    deltatime: f32,
    width: u32,
    height: u32,
    monitor_width: u32,
    monitor_height: u32,
    march_iterations: u32,
    t_max: f32,
    dt_min: f32,
    voxel_grid_side: u32,
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

    march_iterations: u32 = 512,
    t_max: f32 = 50.0,
    dt_min: f32 = 0.0001,
    voxels: struct {
        side: u32 = 100,
        max_side: u32 = 500,
    } = .{},

    frame: u32 = 0,
    time: f32 = 0,
    deltatime: f32 = 0,

    pause_t: bool = false,

    background_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#271212ff"),

    rng: std.Random.Xoshiro256,

    pub fn init(window: *Engine.Window) !@This() {
        const mouse = window.poll_mouse();
        const sze = try window.get_res();

        const rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

        return .{
            .monitor_rez = .{ .width = sze.width, .height = sze.height },
            .camera = math.Camera.init(Vec4{ .z = 5 }, math.Camera.constants.basis.opengl),
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

        if (!self.pause_t) {}
    }

    pub fn uniforms(self: *const @This(), window: *Engine.Window) Uniforms {
        const rot = self.camera.rot_quat();

        const fwd = rot.rotate_vector(self.camera.basis.fwd);
        const right = rot.rotate_vector(self.camera.basis.right);
        const up = rot.rotate_vector(self.camera.basis.up);
        const eye = self.camera.pos;

        return .{
            .world_to_screen = self.camera.world_to_screen_mat(window.extent.width, window.extent.height),
            .eye = eye,
            .fwd = fwd,
            .right = right,
            .up = up,
            .mouse = .{
                .x = self.mouse.x,
                .y = self.mouse.y,
                .left = @intCast(@intFromBool(self.mouse.left)),
                .right = @intCast(@intFromBool(self.mouse.right)),
            },
            .background_color = self.background_color,
            .frame = self.frame,
            .time = self.time,
            .deltatime = self.deltatime,
            .width = window.extent.width,
            .height = window.extent.height,
            .monitor_width = self.monitor_rez.width,
            .monitor_height = self.monitor_rez.height,
            .march_iterations = self.march_iterations,
            .t_max = self.t_max,
            .dt_min = self.dt_min,
            .voxel_grid_side = self.voxels.side,
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

        _ = c.ImGui_ColorEdit3("Background Color", @ptrCast(&state.background_color), c.ImGuiColorEditFlags_Float);

        _ = c.ImGui_Checkbox("Pause t (pause_t)", &state.pause_t);

        _ = c.ImGui_SliderInt("March iterations", @ptrCast(&state.march_iterations), 0, 1024);
        _ = c.ImGui_SliderFloat("t max", &state.t_max, 0.1, 1000.0);
        var pow = @log10(state.dt_min);
        _ = c.ImGui_SliderFloat("dt min (10^this)", &pow, -10.0, 0.0);
        state.dt_min = std.math.pow(f32, 10.0, pow);
    }
};
