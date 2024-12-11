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

const Renderer = @import("renderer.zig");

const state = @import("state.zig");
const AppState = state.AppState;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

pub fn main() !void {
    {
        var engine = try Engine.init();
        defer engine.deinit();

        std.debug.print("using device: {s}\n", .{engine.graphics.props.device_name});

        var gui_engine = try GuiEngine.init(engine.window);
        defer gui_engine.deinit();

        var app_state = AppState.init(engine.window);
        var gui_state = state.GuiState{};

        var renderer = try Renderer.init(&engine, &app_state);
        defer renderer.deinit(&engine.graphics.device);

        var gui_renderer = try GuiEngine.GuiRenderer.init(&engine, &renderer.swapchain);
        defer gui_renderer.deinit(&engine.graphics.device);

        var timer = try std.time.Timer.start();
        while (!engine.window.should_close()) {
            defer engine.window.tick();

            if (engine.window.is_minimized()) {
                continue;
            }

            const lap = timer.lap();
            app_state.tick(lap, engine.window);
            renderer.uniforms = app_state.uniforms(engine.window);

            gui_renderer.render_start();
            gui_state.tick(&app_state, lap);
            try gui_renderer.render_end(&engine.graphics.device, &renderer);

            // multiple framebuffers => multiple descriptor sets => different buffers
            // big buffers that depends on the last frame's big buffer + multiple framebuffers => me sad
            // so just wait for one frame's queue to be empty before trying to render another frame
            try engine.graphics.device.queueWaitIdle(engine.graphics.graphics_queue.handle);

            const present = try renderer.present(&gui_renderer, &engine.graphics);
            // IDK: this never triggers :/
            if (present == .suboptimal) {
                std.debug.print("{any}\n", .{present});
            }

            if (engine.window.resize_fuse.unfuse() or present == .suboptimal) {
                renderer.deinit(&engine.graphics.device);
                renderer = try Renderer.init(&engine, &app_state);

                gui_renderer.deinit(&engine.graphics.device);
                gui_renderer = try GuiEngine.GuiRenderer.init(&engine, &renderer.swapchain);
            }
        }

        try renderer.swapchain.waitForAllFences(&engine.graphics.device);
        try engine.graphics.device.deviceWaitIdle();
    }

    // no defer cuz we don't want to print leaks when we error out
    _ = gpa.deinit();
}
