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

const application = @import("eyeface.zig");
// const application = @import("mandlebulb.zig");
const App = application.App;
const AppState = application.AppState;
const GuiState = application.GuiState;
const RendererState = application.RendererState;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

pub fn main() !void {
    {
        var engine = try Engine.init();
        defer engine.deinit();

        std.debug.print("using device: {s}\n", .{engine.graphics.props.device_name});

        var gui_engine = try GuiEngine.init(engine.window);
        defer gui_engine.deinit();

        var app_state = try AppState.init(engine.window);
        defer app_state.deinit();

        var gui_state = GuiState{};

        var app = try App.init(&engine, &app_state);
        defer app.deinit(&engine.graphics.device);

        var renderer_state = try RendererState.init(&app, &engine, &app_state);
        defer renderer_state.deinit(&engine.graphics.device);

        var gui_renderer = try GuiEngine.GuiRenderer.init(&engine, &renderer_state.swapchain);
        defer gui_renderer.deinit(&engine.graphics.device);

        var timer = try std.time.Timer.start();
        while (!engine.window.should_close()) {
            defer engine.window.tick();

            if (engine.window.is_minimized()) {
                continue;
            }

            const frametime = @as(f32, @floatFromInt(timer.read())) / std.time.ns_per_ms;
            const min_frametime = 1.0 / @as(f32, @floatFromInt(app_state.fps_cap)) * std.time.ms_per_s;
            if (frametime < min_frametime) {
                std.time.sleep(@intFromFloat(std.time.ns_per_ms * (min_frametime - frametime)));
            }

            const lap = timer.lap();
            try app_state.tick(lap, &engine, &app);

            gui_renderer.render_start();
            gui_state.tick(&app_state, lap);
            try gui_renderer.render_end(&engine.graphics.device, &renderer_state.swapchain);

            app.uniforms.uniform_buffer = try app_state.uniforms(engine.window);

            // multiple framebuffers => multiple descriptor sets => different buffers
            // big buffers that depends on the last frame's big buffer + multiple framebuffers => me sad
            // so just wait for one frame's queue to be empty before trying to render another frame
            try engine.graphics.device.queueWaitIdle(engine.graphics.graphics_queue.handle);

            const present = try app.present(&renderer_state, &gui_renderer, &engine.graphics);
            // IDK: this never triggers :/
            if (present == .suboptimal) {
                std.debug.print("{any}\n", .{present});
            }

            if (engine.window.resize_fuse.unfuse() or
                present == .suboptimal or
                app.stages.update() or
                app_state.reset_render_state.unfuse())
            {
                renderer_state.deinit(&engine.graphics.device);
                renderer_state = try RendererState.init(&app, &engine, &app_state);

                gui_renderer.deinit(&engine.graphics.device);
                gui_renderer = try GuiEngine.GuiRenderer.init(&engine, &renderer_state.swapchain);
            }
        }

        try renderer_state.swapchain.waitForAllFences(&engine.graphics.device);
        try engine.graphics.device.deviceWaitIdle();
    }

    // no defer cuz we don't want to print leaks when we error out
    _ = gpa.deinit();
}
