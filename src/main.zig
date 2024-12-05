const std = @import("std");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "1");
    // @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("GLFW/glfw3.h");
    @cInclude("GLFW/glfw3native.h");

    @cInclude("dcimgui.h");
    @cInclude("cimgui_impl_glfw.h");
    @cInclude("cimgui_impl_vulkan.h");
});

const vk = @import("vulkan");

const utils = @import("utils.zig");
const Fuse = utils.Fuse;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

const Engine = struct {
    window: *Window,
    graphics: VulkanContext,

    fn init() !@This() {
        var window = try Window.init();
        errdefer window.deinit();

        var ctx = try VulkanContext.init(window);
        errdefer ctx.deinit();

        return .{
            .window = window,
            .graphics = ctx,
        };
    }

    fn deinit(self: *@This()) void {
        self.graphics.deinit();
        self.window.deinit();
    }

    const Window = struct {
        // last known size
        extent: vk.Extent2D = .{ .width = 800, .height = 600 },
        handle: *c.GLFWwindow,
        resize_fuse: Fuse = .{},

        const Event = union(enum) {
            Resize: struct {
                width: u32,
                height: u32,
            },
        };

        const Callbacks = struct {
            var global_window: *Window = undefined;

            fn err(code: c_int, msg: [*c]const u8) callconv(.C) void {
                _ = code;
                std.debug.print("GLFW Error: {s}\n", .{msg});
            }

            fn resize(_: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
                global_window.extent = .{ .width = @intCast(width), .height = @intCast(height) };
                _ = global_window.resize_fuse.fuse();
            }
        };

        fn init() !*@This() {
            _ = c.glfwSetErrorCallback(&Callbacks.err);

            if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
            errdefer c.glfwTerminate();

            if (c.glfwVulkanSupported() != c.GLFW_TRUE) {
                return error.VulkanNotSupported;
            }

            const extent = .{ .width = 800, .height = 600 };
            c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
            const window = c.glfwCreateWindow(
                extent.width,
                extent.height,
                "yaaaaaaaaaa",
                null,
                null,
            ) orelse return error.WindowInitFailed;
            errdefer c.glfwDestroyWindow(window);

            _ = c.glfwSetFramebufferSizeCallback(window, &Callbacks.resize);

            const self = try allocator.create(@This());
            errdefer allocator.destroy(self);
            self.* = .{
                .extent = extent,
                .handle = window,
            };
            Callbacks.global_window = self;
            return self;
        }

        fn tick(self: *@This()) void {
            var w: c_int = undefined;
            var h: c_int = undefined;
            c.glfwGetFramebufferSize(self.handle, &w, &h);

            if (c.glfwGetKey(self.handle, c.GLFW_KEY_ESCAPE) == c.GLFW_PRESS) {
                c.glfwSetWindowShouldClose(self.handle, c.GL_TRUE);
            }

            // polls events and calls callbacks
            c.glfwPollEvents();
        }

        fn is_pressed(self: *@This(), key: c_int) bool {
            return c.glfwGetKey(self.handle, key) == c.GLFW_PRESS;
        }

        fn poll_mouse(self: *@This()) struct { left: bool, x: i32, y: i32 } {
            var x: f64 = 0;
            var y: f64 = 0;
            c.glfwGetCursorPos(self.handle, @ptrCast(&x), @ptrCast(&y));
            const left = c.glfwGetMouseButton(self.handle, c.GLFW_MOUSE_BUTTON_LEFT);
            return .{ .left = left == c.GLFW_PRESS, .x = @intFromFloat(x), .y = @intFromFloat(y) };
        }

        fn should_close(self: *@This()) bool {
            return c.glfwWindowShouldClose(self.handle) == c.GLFW_TRUE;
        }

        fn queue_close(self: *@This()) void {
            c.glfwSetWindowShouldClose(self.handle, c.GLFW_TRUE);
        }

        fn is_minimized(self: *@This()) bool {
            var w: c_int = undefined;
            var h: c_int = undefined;
            c.glfwGetFramebufferSize(self.handle, &w, &h);
            return w == 0 or h == 0;
        }

        fn deinit(self: *@This()) void {
            c.glfwDestroyWindow(self.handle);
            c.glfwTerminate();
            allocator.destroy(self);
        }
    };

    const VulkanContext = struct {
        const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};
        const Api = struct {
            const apis: []const vk.ApiInfo = &.{
                // .{
                //     .base_commands = .{
                //         .createInstance = true,
                //     },
                //     .instance_commands = .{
                //         .createDevice = true,
                //     },
                // },
                vk.features.version_1_0,
                vk.features.version_1_1,
                vk.features.version_1_2,
                vk.features.version_1_3,
                vk.extensions.khr_surface,
                vk.extensions.khr_swapchain,

                // EH: ?what are these
                // vk.extensions.ext_validation_features,
                // vk.extensions.ext_validation_flags,
                // vk.extensions.ext_validation_cache,
            };

            const BaseDispatch = vk.BaseWrapper(apis);
            const InstanceDispatch = vk.InstanceWrapper(apis);
            const DeviceDispatch = vk.DeviceWrapper(apis);

            const Instance = vk.InstanceProxy(apis);
            const Device = vk.DeviceProxy(apis);
            const CommandBuffer = vk.CommandBufferProxy(apis);
            const Queue = vk.QueueProxy(apis);
        };

        vkb: Api.BaseDispatch,
        instance: Api.Instance,
        surface: vk.SurfaceKHR,
        pdev: vk.PhysicalDevice,
        props: vk.PhysicalDeviceProperties,
        mem_props: vk.PhysicalDeviceMemoryProperties,

        device: Api.Device,
        graphics_queue: Queue,
        present_queue: Queue,

        fn init(window: *Window) !@This() {
            const vkb = try Api.BaseDispatch.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(&c.glfwGetInstanceProcAddress)));

            var glfw_exts_count: u32 = 0;
            const glfw_exts = c.glfwGetRequiredInstanceExtensions(&glfw_exts_count);
            const layers = [_][*c]const u8{
                "VK_LAYER_KHRONOS_validation",
            };
            const instance = vkb.createInstance(&.{
                .p_application_info = &.{
                    .p_application_name = "yaaaaaaaaaaaaaaa",
                    .api_version = vk.API_VERSION_1_3,
                    .application_version = vk.makeApiVersion(0, 0, 1, 0),
                    .engine_version = vk.makeApiVersion(0, 0, 1, 0),
                },
                .enabled_extension_count = glfw_exts_count,
                .pp_enabled_extension_names = @ptrCast(glfw_exts),
                .enabled_layer_count = @intCast(layers.len),
                .pp_enabled_layer_names = @ptrCast(&layers),
            }, null) catch |e| {
                std.debug.print("{any}\n", .{e});
                return e;
            };

            const vki = try allocator.create(Api.InstanceDispatch);
            errdefer allocator.destroy(vki);
            vki.* = try Api.InstanceDispatch.load(instance, vkb.dispatch.vkGetInstanceProcAddr);
            const vkinstance: Api.Instance = Api.Instance.init(instance, vki);
            errdefer vkinstance.destroyInstance(null);

            var surface: vk.SurfaceKHR = undefined;
            if (c.glfwCreateWindowSurface(@as(*const c.VkInstance, @ptrCast(&vkinstance.handle)).*, window.handle, null, @ptrCast(&surface)) != c.VK_SUCCESS) {
                return error.SurfaceInitFailed;
            }

            const candidate = try DeviceCandidate.pick(vkinstance, surface);
            const pdev = candidate.pdev;
            const props = candidate.props;

            const dev = blk: {
                const priority = [_]f32{1};
                const qci = [_]vk.DeviceQueueCreateInfo{
                    .{
                        .queue_family_index = candidate.queues.graphics_family,
                        .queue_count = 1,
                        .p_queue_priorities = &priority,
                    },
                    .{
                        .queue_family_index = candidate.queues.present_family,
                        .queue_count = 1,
                        .p_queue_priorities = &priority,
                    },
                };

                const queue_count: u32 = if (candidate.queues.graphics_family == candidate.queues.present_family)
                    1
                else
                    2;

                const device = try vkinstance.createDevice(candidate.pdev, &.{
                    .queue_create_info_count = queue_count,
                    .p_queue_create_infos = &qci,
                    .enabled_extension_count = required_device_extensions.len,
                    .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
                    .p_enabled_features = &.{
                        .fill_mode_non_solid = vk.TRUE,
                        // .vertex_pipeline_stores_and_atomics = vk.TRUE,
                    },
                }, null);
                break :blk device;
            };

            const vkd = try allocator.create(Api.DeviceDispatch);
            errdefer allocator.destroy(vkd);
            vkd.* = try Api.DeviceDispatch.load(dev, vkinstance.wrapper.dispatch.vkGetDeviceProcAddr);
            const device = Api.Device.init(dev, vkd);
            errdefer device.destroyDevice(null);

            const graphics_queue = Queue.init(device, candidate.queues.graphics_family);
            const present_queue = Queue.init(device, candidate.queues.present_family);

            const mem_props = vkinstance.getPhysicalDeviceMemoryProperties(pdev);

            return .{
                .vkb = vkb,
                .instance = vkinstance,
                .surface = surface,
                .device = device,
                .pdev = pdev,
                .props = props,
                .mem_props = mem_props,
                .graphics_queue = graphics_queue,
                .present_queue = present_queue,
            };
        }

        fn deinit(self: *@This()) void {
            self.device.destroyDevice(null);
            self.instance.destroySurfaceKHR(self.surface, null);
            self.instance.destroyInstance(null);

            // Don't forget to free the tables to prevent a memory leak.
            allocator.destroy(self.device.wrapper);
            allocator.destroy(self.instance.wrapper);
        }

        pub fn allocate(self: @This(), requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
            return try self.device.allocateMemory(&.{
                .allocation_size = requirements.size,
                .memory_type_index = blk: {
                    for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
                        if (requirements.memory_type_bits & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(flags)) {
                            break :blk @truncate(i);
                        }
                    }

                    return error.NoSuitableMemoryType;
                },
            }, null);
        }

        pub const Queue = struct {
            handle: vk.Queue,
            family: u32,

            fn init(device: Api.Device, family: u32) Queue {
                return .{
                    .handle = device.getDeviceQueue(family, 0),
                    .family = family,
                };
            }
        };

        const DeviceCandidate = struct {
            pdev: vk.PhysicalDevice,
            props: vk.PhysicalDeviceProperties,
            queues: QueueAllocation,

            const QueueAllocation = struct {
                graphics_family: u32,
                present_family: u32,
            };

            fn pick(
                instance: Api.Instance,
                surface: vk.SurfaceKHR,
            ) !DeviceCandidate {
                const pdevs = try instance.enumeratePhysicalDevicesAlloc(allocator);
                defer allocator.free(pdevs);

                for (pdevs) |pdev| {
                    const props = instance.getPhysicalDeviceProperties(pdev);
                    std.debug.print("{s}\n", .{props.device_name});
                }

                for (pdevs) |pdev| {
                    if (try checkSuitable(instance, pdev, surface)) |candidate| {
                        return candidate;
                    }
                }

                return error.NoSuitableDevice;
            }

            fn checkSuitable(
                instance: Api.Instance,
                pdev: vk.PhysicalDevice,
                surface: vk.SurfaceKHR,
            ) !?DeviceCandidate {
                const propsv = try instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, allocator);
                defer allocator.free(propsv);
                for (required_device_extensions) |ext| {
                    for (propsv) |props| {
                        if (std.mem.eql(u8, std.mem.span(ext), std.mem.sliceTo(&props.extension_name, 0))) {
                            break;
                        }
                    } else {
                        return null;
                    }
                }

                var format_count: u32 = undefined;
                _ = try instance.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);
                var present_mode_count: u32 = undefined;
                _ = try instance.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);
                if (!(format_count > 0 and present_mode_count > 0)) {
                    return null;
                }

                const families = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, allocator);
                defer allocator.free(families);

                var graphics_family: ?u32 = null;
                var present_family: ?u32 = null;
                for (families, 0..) |properties, i| {
                    const family: u32 = @intCast(i);
                    if (graphics_family == null and properties.queue_flags.graphics_bit) {
                        graphics_family = family;
                    }
                    if (present_family == null and (try instance.getPhysicalDeviceSurfaceSupportKHR(pdev, family, surface)) == vk.TRUE) {
                        present_family = family;
                    }
                }

                if (graphics_family == null or present_family == null) {
                    return null;
                }

                const props = instance.getPhysicalDeviceProperties(pdev);
                if (props.device_type != .discrete_gpu) {
                    return null;
                }
                return DeviceCandidate{
                    .pdev = pdev,
                    .props = props,
                    .queues = .{
                        .graphics_family = graphics_family.?,
                        .present_family = present_family.?,
                    },
                };
            }
        };
    };
};

const Renderer = struct {
    swapchain: Swapchain,
    uniforms: Uniforms,
    uniform_buffer: vk.Buffer,
    uniform_memory: vk.DeviceMemory,
    descriptor_pool: vk.DescriptorPool,
    frag_descriptor_set_layout: vk.DescriptorSetLayout,
    frag_descriptor_set: vk.DescriptorSet,
    compute_descriptor_set_layout: vk.DescriptorSetLayout,
    compute_descriptor_set: vk.DescriptorSet,
    pass: vk.RenderPass,
    compute_pipeline_layout: vk.PipelineLayout,
    compute_pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,
    // framebuffers are objects containing views of swapchain images
    framebuffers: []vk.Framebuffer,
    command_pool: vk.CommandPool,
    vertex_buffer_memory: vk.DeviceMemory,
    vertex_buffer: vk.Buffer,
    // command buffers are recordings of a bunch of commands that a gpu can execute
    command_buffers: []vk.CommandBuffer,

    const Device = Engine.VulkanContext.Api.Device;
    const Vertex = extern struct {
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

    pub const transformers = struct {
        pub fn TransformSet(n: u32) type {
            return extern struct {
                transforms: [n]utils.Mat4x4 = std.mem.zeroes([n]utils.Mat4x4),

                const Builder = struct {
                    transforms: [n]Transforms,

                    const Transforms = struct {
                        translate: utils.Mat4x4,
                        shear: utils.Mat4x4,
                        scale: utils.Mat4x4,
                        rot: utils.Mat4x4,

                        fn init() @This() {
                            return .{
                                .translate = utils.Mat4x4.identity(),
                                .shear = utils.Mat4x4.identity(),
                                .scale = utils.Mat4x4.identity(),
                                .rot = utils.Mat4x4.identity(),
                            };
                        }

                        fn random(_rng: std.Random) @This() {
                            const rng = utils.Rng.init(_rng);

                            const scale = utils.Mat4x4.random.scale(&rng.with(.{ .min = 0.4, .max = 0.7, .flip_sign = false }));
                            const translate = utils.Mat4x4.random.translate(&rng);
                            const rot = utils.Mat4x4.random.rot(&rng);
                            const shear = utils.Mat4x4.random.shear(&rng.with(.{ .min = 0.0, .max = 0.3 }));

                            return .{
                                .translate = translate,
                                .shear = shear,
                                .scale = scale,
                                .rot = rot,
                            };
                        }

                        fn combine(self: *const @This()) utils.Mat4x4 {
                            var mat = utils.Mat4x4.identity();
                            mat = mat.mul_mat(self.translate);
                            mat = mat.mul_mat(self.shear);
                            mat = mat.mul_mat(self.scale);
                            mat = mat.mul_mat(self.rot);
                            return mat;
                        }

                        fn mix(self: *const @This(), other: *const @This(), t: f32) @This() {
                            return .{
                                .translate = self.translate.mix(&other.translate, t),
                                .shear = self.shear.mix(&other.shear, t),
                                .scale = self.scale.mix(&other.scale, t),
                                .rot = self.rot.mix(&other.rot, t),
                            };
                        }
                    };

                    fn init() @This() {
                        var this = std.mem.zeroes(@This());

                        inline for (0..n) |i| {
                            this.transforms[i] = Transforms.init();
                        }

                        return this;
                    }

                    fn random() @This() {
                        var _rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
                        const rng = _rng.random();

                        var this = std.mem.zeroes(@This());

                        inline for (0..n) |i| {
                            this.transforms[i] = Transforms.random(rng);
                        }

                        return this;
                    }

                    fn build(self: *const @This()) TransformSet(n) {
                        var this = std.mem.zeroes(TransformSet(n));

                        inline for (0..n) |i| {
                            this.transforms[i] = self.transforms[i].combine();
                        }

                        return this;
                    }
                };
            };
        }

        pub fn sirpinski_pyramid() TransformSet(5).Builder {
            var this = TransformSet(5).Builder.init();

            const s = utils.Mat4x4.scaling_mat(utils.Vec4.splat3(0.5));

            inline for (0..5) |i| {
                this.transforms[i].scale = s;
            }

            this.transforms[0].translate = utils.Mat4x4.translation_mat(.{ .x = 0.0, .y = 1.0, .z = 0.0 });
            this.transforms[1].translate = utils.Mat4x4.translation_mat(.{ .x = 1.0, .y = -1.0, .z = 1.0 });
            this.transforms[2].translate = utils.Mat4x4.translation_mat(.{ .x = 1.0, .y = -1.0, .z = -1.0 });
            this.transforms[3].translate = utils.Mat4x4.translation_mat(.{ .x = -1.0, .y = -1.0, .z = 1.0 });
            this.transforms[4].translate = utils.Mat4x4.translation_mat(.{ .x = -1.0, .y = -1.0, .z = -1.0 });

            return this;
        }
    };

    const Uniforms = extern struct {
        transforms: TransformSet,
        view_matrix: utils.Mat4x4,
        projection_matrix: utils.Mat4x4,
        world_to_screen: utils.Mat4x4,
        eye: utils.Vec4,
        mouse: extern struct { x: i32 = 0, y: i32 = 0, left: u32 = 0, right: u32 = 0 } = .{},
        pitch: f32 = 0,
        yaw: f32 = 0,
        frame: u32 = 0,

        const TransformSet = transformers.TransformSet(5);

        fn init(width: u32, height: u32) @This() {
            const eye = utils.Vec4{ .z = -5 };
            const up = utils.Vec4{ .y = -1 };
            const at = utils.Vec4{ .z = 1 };
            const view = utils.Mat4x4.view(eye, at, up);
            const projection = utils.Mat4x4.perspective_projection(height, width, 0.01, 100.0, std.math.pi / 3.0);

            const transforms: TransformSet = TransformSet.Builder.random().build();

            return .{
                .eye = eye,
                .view_matrix = view,
                .projection_matrix = projection,
                .world_to_screen = projection.mul_mat(view),
                .transforms = transforms,
            };
        }

        fn tick(self: *@This(), timer: *std.time.Timer, window: *Engine.Window) void {
            const pitch_min = -std.math.pi / 2.0 + 0.1;
            const pitch_max = std.math.pi / 2.0 - 0.1;
            var up = utils.Vec4{ .y = -1 };
            var fwd = utils.Vec4{ .z = 1 };
            var right = utils.Vec4{ .x = 1 };
            const speed = 1.0;
            const sensitivity = 1.0;

            const lap = timer.lap();
            const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));
            std.debug.print("fps: {d}\n", .{@as(u32, @intFromFloat(1.0 / delta))});
            const w = window.is_pressed(c.GLFW_KEY_W);
            const a = window.is_pressed(c.GLFW_KEY_A);
            const s = window.is_pressed(c.GLFW_KEY_S);
            const d = window.is_pressed(c.GLFW_KEY_D);
            const mouse = window.poll_mouse();

            if (mouse.left) {
                self.yaw += @as(f32, @floatFromInt(mouse.x - self.mouse.x)) * sensitivity * delta;
                self.pitch -= @as(f32, @floatFromInt(mouse.y - self.mouse.y)) * sensitivity * delta;
                self.pitch = std.math.clamp(self.pitch, pitch_min, pitch_max);
            }

            self.mouse.left = @intCast(@intFromBool(mouse.left));
            self.mouse.x = mouse.x;
            self.mouse.y = mouse.y;

            var rot = utils.Vec4.quat_identity_rot();
            rot = rot.quat_mul(utils.Vec4.quat_angle_axis(self.pitch, right));
            rot = rot.quat_mul(utils.Vec4.quat_angle_axis(self.yaw, up));
            rot = rot.quat_conjugate();

            up = rot.rotate_vector(up);
            fwd = rot.rotate_vector(fwd);
            right = rot.rotate_vector(right);

            if (w) {
                self.eye = self.eye.add(fwd.scale(delta * speed));
            }
            if (a) {
                self.eye = self.eye.sub(right.scale(delta * speed));
            }
            if (s) {
                self.eye = self.eye.sub(fwd.scale(delta * speed));
            }
            if (d) {
                self.eye = self.eye.add(right.scale(delta * speed));
            }

            const projection = utils.Mat4x4.perspective_projection(window.extent.height, window.extent.width, 0.01, 100.0, std.math.pi / 3.0);
            self.projection_matrix = projection;
            self.view_matrix = utils.Mat4x4.view(self.eye, fwd, up);
            self.world_to_screen = self.projection_matrix.mul_mat(self.view_matrix);
            self.frame += 1;
        }
    };

    fn init(engine: *Engine) !@This() {
        var ctx = &engine.graphics;
        const device = &ctx.device;

        var swapchain = try Swapchain.init(ctx, engine.window.extent);
        errdefer swapchain.deinit(device);

        const uniform_buffer = try device.createBuffer(&.{
            .size = @sizeOf(Uniforms),
            .usage = .{
                .uniform_buffer_bit = true,
            },
            .sharing_mode = .exclusive,
        }, null);
        errdefer device.destroyBuffer(uniform_buffer, null);
        const uniform_mem_req = device.getBufferMemoryRequirements(uniform_buffer);
        const uniform_buffer_memory = try ctx.allocate(uniform_mem_req, .{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        });
        errdefer device.freeMemory(uniform_buffer_memory, null);
        try device.bindBufferMemory(uniform_buffer, uniform_buffer_memory, 0);

        const pool = try device.createCommandPool(&.{
            .queue_family_index = ctx.graphics_queue.family,
            .flags = .{
                .reset_command_buffer_bit = true,
            },
        }, null);
        errdefer device.destroyCommandPool(pool, null);

        var vertices = std.ArrayList(Vertex).init(allocator);
        defer vertices.deinit();
        try vertices.appendNTimes(.{ .pos = .{ 0, 0, 0, 1 } }, 64 * 80000);

        const vertex_buffer = blk: {
            const buffer = try device.createBuffer(&.{
                .size = @sizeOf(Vertex) * vertices.items.len,
                .usage = .{
                    .transfer_dst_bit = true,
                    .vertex_buffer_bit = true,
                    .storage_buffer_bit = true,
                },
                .sharing_mode = .exclusive,
            }, null);
            errdefer device.destroyBuffer(buffer, null);
            const mem_reqs = device.getBufferMemoryRequirements(buffer);
            const memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
            errdefer device.freeMemory(memory, null);
            try device.bindBufferMemory(buffer, memory, 0);

            const staging_buffer = try device.createBuffer(&.{
                .size = @sizeOf(Vertex) * vertices.items.len,
                .usage = .{ .transfer_src_bit = true },
                .sharing_mode = .exclusive,
            }, null);
            defer device.destroyBuffer(staging_buffer, null);
            const staging_mem_reqs = device.getBufferMemoryRequirements(staging_buffer);
            const staging_memory = try ctx.allocate(staging_mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
            defer device.freeMemory(staging_memory, null);
            try device.bindBufferMemory(staging_buffer, staging_memory, 0);

            {
                const data = try device.mapMemory(staging_memory, 0, vk.WHOLE_SIZE, .{});
                defer device.unmapMemory(staging_memory);

                const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
                @memcpy(gpu_vertices, vertices.items);
            }

            try copyBuffer(ctx, device, pool, buffer, staging_buffer, @sizeOf(Vertex) * vertices.items.len);

            break :blk .{ .buffer = buffer, .memory = memory };
        };
        errdefer device.destroyBuffer(vertex_buffer.buffer, null);
        errdefer device.freeMemory(vertex_buffer.memory, null);

        const frag_desc_set_layout = try device.createDescriptorSetLayout(&.{
            .flags = .{},
            .binding_count = 1,
            .p_bindings = &[_]vk.DescriptorSetLayoutBinding{
                .{
                    .binding = 0,
                    .descriptor_type = .uniform_buffer,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .vertex_bit = true,
                        .fragment_bit = true,
                        .compute_bit = true,
                    },
                },
            },
        }, null);
        errdefer device.destroyDescriptorSetLayout(frag_desc_set_layout, null);
        const compute_desc_set_layout = try device.createDescriptorSetLayout(&.{
            .flags = .{},
            .binding_count = 2,
            .p_bindings = &[_]vk.DescriptorSetLayoutBinding{
                .{
                    .binding = 0,
                    .descriptor_type = .uniform_buffer,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .vertex_bit = true,
                        .fragment_bit = true,
                        .compute_bit = true,
                    },
                },
                .{
                    .binding = 1,
                    .descriptor_type = .storage_buffer,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .vertex_bit = true,
                        .fragment_bit = true,
                        .compute_bit = true,
                    },
                },
            },
        }, null);
        errdefer device.destroyDescriptorSetLayout(compute_desc_set_layout, null);
        const desc_pool = try device.createDescriptorPool(&.{
            .max_sets = 2,
            .pool_size_count = 2,
            .p_pool_sizes = &[_]vk.DescriptorPoolSize{
                .{
                    .type = .uniform_buffer,
                    .descriptor_count = 1,
                },
                .{
                    .type = .storage_buffer,
                    .descriptor_count = 1,
                },
            },
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
        device.updateDescriptorSets(1, &[_]vk.WriteDescriptorSet{.{
            .dst_set = frag_desc_set,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .uniform_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = uniform_buffer,
                .offset = 0,
                .range = @sizeOf(Uniforms),
            }},

            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        }}, 0, null);
        device.updateDescriptorSets(2, &[_]vk.WriteDescriptorSet{
            .{
                .dst_set = compute_desc_set,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .uniform_buffer,
                .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                    .buffer = uniform_buffer,
                    .offset = 0,
                    .range = @sizeOf(Uniforms),
                }},
                // OOF: ??
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = compute_desc_set,
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                    .buffer = vertex_buffer.buffer,
                    .offset = 0,
                    .range = vk.WHOLE_SIZE,
                }},
                // OOF: ??
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
        }, 0, null);

        var compiler = utils.Glslc.Compiler{ .opt = .fast, .env = .vulkan1_3 };
        const vert_spv = blk: {
            compiler.stage = .vertex;
            const vert: utils.Glslc.Compiler.Code = .{ .path = .{
                .main = "./src/vert.glsl",
                .include = &[_][]const u8{},
                .definitions = &[_][]const u8{},
            } };
            // _ = compiler.dump_assembly(allocator, &vert);
            const res = try compiler.compile(
                allocator,
                &vert,
                .spirv,
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
            compiler.stage = .fragment;
            const frag: utils.Glslc.Compiler.Code = .{ .path = .{
                .main = "./src/frag.glsl",
                .include = &[_][]const u8{},
                .definitions = &[_][]const u8{},
            } };
            // _ = compiler.dump_assembly(allocator, &frag);
            const res = try compiler.compile(
                allocator,
                &frag,
                .spirv,
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
        const compute_spv = blk: {
            compiler.stage = .compute;
            const frag: utils.Glslc.Compiler.Code = .{ .path = .{
                .main = "./src/compute.glsl",
                .include = &[_][]const u8{},
                .definitions = &[_][]const u8{},
            } };
            // _ = compiler.dump_assembly(allocator, &frag);
            const res = try compiler.compile(
                allocator,
                &frag,
                .spirv,
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
        defer allocator.free(compute_spv);

        const pass = blk: {
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

            const color_attachment_ref = vk.AttachmentReference{
                .attachment = 0,
                .layout = .color_attachment_optimal,
            };

            const subpass = vk.SubpassDescription{
                .pipeline_bind_point = .graphics,
                .color_attachment_count = 1,
                .p_color_attachments = @ptrCast(&color_attachment_ref),
            };

            break :blk try device.createRenderPass(&.{
                .attachment_count = 1,
                .p_attachments = @ptrCast(&color_attachment),
                .subpass_count = 1,
                .p_subpasses = @ptrCast(&subpass),
            }, null);
        };
        errdefer device.destroyRenderPass(pass, null);

        const compute_pipeline_layout = try device.createPipelineLayout(&.{
            // .flags: PipelineLayoutCreateFlags = .{},
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&compute_desc_set_layout),
        }, null);
        errdefer device.destroyPipelineLayout(compute_pipeline_layout, null);
        const compute_pipeline = blk: {
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

            var pipeline: vk.Pipeline = undefined;
            _ = try device.createComputePipelines(.null_handle, 1, @ptrCast(&cpci), null, @ptrCast(&pipeline));
            break :blk pipeline;
        };
        errdefer device.destroyPipeline(compute_pipeline, null);

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
                .vertex_binding_description_count = 1,
                .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
                .vertex_attribute_description_count = Vertex.attribute_description.len,
                .p_vertex_attribute_descriptions = &Vertex.attribute_description,
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
                .polygon_mode = .point,
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
                .p_depth_stencil_state = null,
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
                fb.* = try device.createFramebuffer(&.{
                    .render_pass = pass,
                    .attachment_count = 1,
                    .p_attachments = @ptrCast(&swapchain.swap_images[i_2].view),
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

        const uniforms = Uniforms.init(engine.window.extent.width, engine.window.extent.height);

        const command_buffers = blk: {
            const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
            errdefer allocator.free(cmdbufs);

            try device.allocateCommandBuffers(&.{
                .command_pool = pool,
                .level = .primary,
                .command_buffer_count = @intCast(cmdbufs.len),
            }, cmdbufs.ptr);
            errdefer device.freeCommandBuffers(pool, @intCast(cmdbufs.len), cmdbufs.ptr);

            const clear = vk.ClearValue{
                .color = .{
                    .float_32 = utils.ColorParse.hex_xyzw(utils.Vec4, "#282828ff").gamma_correct_inv().to_buf(),
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

                device.cmdBindPipeline(cmdbuf, .compute, compute_pipeline);
                device.cmdBindDescriptorSets(cmdbuf, .compute, compute_pipeline_layout, 0, 1, @ptrCast(&compute_desc_set), 0, null);
                device.cmdDispatch(cmdbuf, @intCast(vertices.items.len / 64), 1, 1);

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
                    .clear_value_count = 1,
                    .p_clear_values = @ptrCast(&clear),
                }, .@"inline");

                device.cmdBindPipeline(cmdbuf, .graphics, pipeline);
                device.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&vertex_buffer.buffer), &[_]vk.DeviceSize{0});
                device.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 0, 1, @ptrCast(&frag_desc_set), 0, null);
                device.cmdDraw(cmdbuf, @intCast(vertices.items.len), uniforms.transforms.transforms.len, 0, 0);

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
            .uniform_buffer = uniform_buffer,
            .uniform_memory = uniform_buffer_memory,
            .descriptor_pool = desc_pool,
            .frag_descriptor_set_layout = frag_desc_set_layout,
            .frag_descriptor_set = frag_desc_set,
            .compute_descriptor_set_layout = compute_desc_set_layout,
            .compute_descriptor_set = compute_desc_set,
            .pass = pass,
            .compute_pipeline_layout = compute_pipeline_layout,
            .compute_pipeline = compute_pipeline,
            .pipeline_layout = pipeline_layout,
            .pipeline = pipeline,
            .framebuffers = framebuffers,
            .command_pool = pool,
            .vertex_buffer_memory = vertex_buffer.memory,
            .vertex_buffer = vertex_buffer.buffer,
            .command_buffers = command_buffers,
        };
    }

    fn deinit(self: *@This(), device: *Device) void {
        try self.swapchain.waitForAllFences(device);

        defer self.swapchain.deinit(device);

        defer {
            device.destroyBuffer(self.uniform_buffer, null);
            device.freeMemory(self.uniform_memory, null);
        }
        defer {
            device.destroyBuffer(self.vertex_buffer, null);
            device.freeMemory(self.vertex_buffer_memory, null);
        }
        defer device.destroyDescriptorSetLayout(self.frag_descriptor_set_layout, null);
        defer device.destroyDescriptorSetLayout(self.compute_descriptor_set_layout, null);
        defer device.destroyDescriptorPool(self.descriptor_pool, null);

        defer device.destroyRenderPass(self.pass, null);
        defer device.destroyPipelineLayout(self.compute_pipeline_layout, null);
        defer device.destroyPipeline(self.compute_pipeline, null);
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

    fn present(self: *@This(), gui: *Gui.GuiRenderer, ctx: *Engine.VulkanContext) !Swapchain.PresentState {
        const cmdbuf = self.command_buffers[self.swapchain.image_index];
        const gui_cmdbuf = gui.cmd_bufs[self.swapchain.image_index];

        return self.swapchain.present(&[_]vk.CommandBuffer{ cmdbuf, gui_cmdbuf }, ctx, &self.uniforms, &self.uniform_memory) catch |err| switch (err) {
            error.OutOfDateKHR => return .suboptimal,
            else => |narrow| return narrow,
        };
    }

    fn copyBuffer(ctx: *Engine.VulkanContext, device: *Device, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
        var cmdbuf_handle: vk.CommandBuffer = undefined;
        try device.allocateCommandBuffers(&.{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&cmdbuf_handle));
        defer device.freeCommandBuffers(pool, 1, @ptrCast(&cmdbuf_handle));

        const cmdbuf = Engine.VulkanContext.Api.CommandBuffer.init(cmdbuf_handle, device.wrapper);

        try cmdbuf.beginCommandBuffer(&.{
            .flags = .{ .one_time_submit_bit = true },
        });

        const region = vk.BufferCopy{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        };
        cmdbuf.copyBuffer(src, dst, 1, @ptrCast(&region));

        try cmdbuf.endCommandBuffer();

        const si = vk.SubmitInfo{
            .command_buffer_count = 1,
            .p_command_buffers = (&cmdbuf.handle)[0..1],
            .p_wait_dst_stage_mask = undefined,
        };
        try device.queueSubmit(ctx.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
        try device.queueWaitIdle(ctx.graphics_queue.handle);
    }

    const Swapchain = struct {
        surface_format: vk.SurfaceFormatKHR,
        present_mode: vk.PresentModeKHR,
        extent: vk.Extent2D,
        handle: vk.SwapchainKHR,

        swap_images: []SwapImage,
        image_index: u32,
        next_image_acquired: vk.Semaphore,

        pub const PresentState = enum {
            optimal,
            suboptimal,
        };

        pub fn init(ctx: *Engine.VulkanContext, extent: vk.Extent2D) !Swapchain {
            return try initRecycle(ctx, extent, .null_handle);
        }

        pub fn initRecycle(ctx: *Engine.VulkanContext, extent: vk.Extent2D, old_handle: vk.SwapchainKHR) !Swapchain {
            const caps = try ctx.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(ctx.pdev, ctx.surface);
            const actual_extent = blk: {
                if (caps.current_extent.width != 0xFFFF_FFFF) {
                    break :blk caps.current_extent;
                } else {
                    break :blk vk.Extent2D{
                        .width = std.math.clamp(
                            extent.width,
                            caps.min_image_extent.width,
                            caps.max_image_extent.width,
                        ),
                        .height = std.math.clamp(
                            extent.height,
                            caps.min_image_extent.height,
                            caps.max_image_extent.height,
                        ),
                    };
                }
            };
            if (actual_extent.width == 0 or actual_extent.height == 0) {
                return error.InvalidSurfaceDimensions;
            }

            const surface_format = blk: {
                const preferred = vk.SurfaceFormatKHR{
                    .format = .b8g8r8a8_srgb,
                    .color_space = .srgb_nonlinear_khr,
                };

                const surface_formats = try ctx.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(ctx.pdev, ctx.surface, allocator);
                defer allocator.free(surface_formats);

                for (surface_formats) |sfmt| {
                    if (std.meta.eql(sfmt, preferred)) {
                        break :blk preferred;
                    }
                }

                break :blk surface_formats[0]; // There must always be at least one supported surface format
            };
            const present_mode = blk: {
                const present_modes = try ctx.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(ctx.pdev, ctx.surface, allocator);
                defer allocator.free(present_modes);

                const preferred = [_]vk.PresentModeKHR{
                    .mailbox_khr,
                    .immediate_khr,
                };

                for (preferred) |mode| {
                    if (std.mem.indexOfScalar(vk.PresentModeKHR, present_modes, mode) != null) {
                        break :blk mode;
                    }
                }

                break :blk .fifo_khr;
            };

            var image_count = caps.min_image_count + 1;
            if (caps.max_image_count > 0) {
                image_count = @min(image_count, caps.max_image_count);
            }

            const qfi = [_]u32{ ctx.graphics_queue.family, ctx.present_queue.family };
            const sharing_mode: vk.SharingMode = if (ctx.graphics_queue.family != ctx.present_queue.family)
                .concurrent
            else
                .exclusive;

            const handle = try ctx.device.createSwapchainKHR(&.{
                .surface = ctx.surface,
                .min_image_count = image_count,
                .image_format = surface_format.format,
                .image_color_space = surface_format.color_space,
                .image_extent = actual_extent,
                .image_array_layers = 1,
                .image_usage = .{ .color_attachment_bit = true, .transfer_dst_bit = true },
                .image_sharing_mode = sharing_mode,
                .queue_family_index_count = qfi.len,
                .p_queue_family_indices = &qfi,
                .pre_transform = caps.current_transform,
                .composite_alpha = .{ .opaque_bit_khr = true },
                .present_mode = present_mode,
                .clipped = vk.TRUE,
                .old_swapchain = old_handle,
            }, null);
            errdefer ctx.device.destroySwapchainKHR(handle, null);

            if (old_handle != .null_handle) {
                // Apparently, the old swapchain handle still needs to be destroyed after recreating.
                ctx.device.destroySwapchainKHR(old_handle, null);
            }

            const swap_images = blk: {
                const images = try ctx.device.getSwapchainImagesAllocKHR(handle, allocator);
                defer allocator.free(images);

                const swap_images = try allocator.alloc(SwapImage, images.len);
                errdefer allocator.free(swap_images);

                var i: usize = 0;
                errdefer for (swap_images[0..i]) |si| si.deinit(&ctx.device);
                for (images) |image| {
                    swap_images[i] = try SwapImage.init(&ctx.device, image, surface_format.format);
                    i += 1;
                }

                break :blk swap_images;
            };
            errdefer {
                for (swap_images) |si| si.deinit(&ctx.device);
                allocator.free(swap_images);
            }

            var next_image_acquired = try ctx.device.createSemaphore(&.{}, null);
            errdefer ctx.device.destroySemaphore(next_image_acquired, null);

            const result = try ctx.device.acquireNextImageKHR(handle, std.math.maxInt(u64), next_image_acquired, .null_handle);
            if (result.result != .success) {
                return error.ImageAcquireFailed;
            }

            std.mem.swap(vk.Semaphore, &swap_images[result.image_index].image_acquired, &next_image_acquired);
            return Swapchain{
                .surface_format = surface_format,
                .present_mode = present_mode,
                .extent = actual_extent,
                .handle = handle,
                .swap_images = swap_images,
                .image_index = result.image_index,
                .next_image_acquired = next_image_acquired,
            };
        }

        fn deinitExceptSwapchain(self: Swapchain, device: *Engine.VulkanContext.Api.Device) void {
            for (self.swap_images) |si| si.deinit(device);
            allocator.free(self.swap_images);
            device.destroySemaphore(self.next_image_acquired, null);
        }

        pub fn waitForAllFences(self: Swapchain, device: *Engine.VulkanContext.Api.Device) !void {
            for (self.swap_images) |si| si.waitForFence(device) catch {};
        }

        pub fn deinit(self: Swapchain, device: *Engine.VulkanContext.Api.Device) void {
            self.deinitExceptSwapchain(device);
            device.destroySwapchainKHR(self.handle, null);
        }

        pub fn recreate(self: *Swapchain, new_extent: vk.Extent2D, ctx: *Engine.VulkanContext) !void {
            const old_handle = self.handle;
            self.deinitExceptSwapchain();
            self.* = try initRecycle(ctx, new_extent, old_handle);
        }

        pub fn currentImage(self: Swapchain) vk.Image {
            return self.swap_images[self.image_index].image;
        }

        pub fn currentSwapImage(self: Swapchain) *const SwapImage {
            return &self.swap_images[self.image_index];
        }

        pub fn present(self: *Swapchain, cmdbufs: []const vk.CommandBuffer, ctx: *Engine.VulkanContext, uniforms: *Uniforms, uniform_memory: *vk.DeviceMemory) !PresentState {
            // Simple method:
            // 1) Acquire next image
            // 2) Wait for and reset fence of the acquired image
            // 3) Submit command buffer with fence of acquired image,
            //    dependendent on the semaphore signalled by the first step.
            // 4) Present current frame, dependent on semaphore signalled by previous step
            // Problem: This way we can't reference the current image while rendering.
            // Better method: Shuffle the steps around such that acquire next image is the last step,
            // leaving the swapchain in a state with the current image.
            // 1) Wait for and reset fence of current image
            // 2) Submit command buffer, signalling fence of current image and dependent on
            //    the semaphore signalled by step 4.
            // 3) Present current frame, dependent on semaphore signalled by the submit
            // 4) Acquire next image, signalling its semaphore
            // One problem that arises is that we can't know beforehand which semaphore to signal,
            // so we keep an extra auxilery semaphore that is swapped around

            // Step 1: Make sure the current frame has finished rendering
            const current = self.currentSwapImage();
            try current.waitForFence(&ctx.device);
            try ctx.device.resetFences(1, @ptrCast(&current.frame_fence));

            {
                const maybe_mapped = try ctx.device.mapMemory(uniform_memory.*, 0, @sizeOf(@TypeOf(uniforms.*)), .{});
                const mapped = maybe_mapped orelse return error.MappingMemoryFailed;
                defer ctx.device.unmapMemory(uniform_memory.*);

                @memcpy(@as([*]u8, @ptrCast(mapped)), std.mem.asBytes(uniforms));
            }

            // Step 2: Submit the command buffer
            const wait_stage = [_]vk.PipelineStageFlags{.{ .top_of_pipe_bit = true }};
            try ctx.device.queueSubmit(ctx.graphics_queue.handle, 1, &[_]vk.SubmitInfo{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&current.image_acquired),
                .p_wait_dst_stage_mask = &wait_stage,
                .command_buffer_count = @intCast(cmdbufs.len),
                .p_command_buffers = cmdbufs.ptr,
                .signal_semaphore_count = 1,
                .p_signal_semaphores = @ptrCast(&current.render_finished),
            }}, current.frame_fence);

            // Step 3: Present the current frame
            _ = try ctx.device.queuePresentKHR(ctx.present_queue.handle, &.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&current.render_finished),
                .swapchain_count = 1,
                .p_swapchains = @ptrCast(&self.handle),
                .p_image_indices = @ptrCast(&self.image_index),
            });

            // Step 4: Acquire next frame
            const result = try ctx.device.acquireNextImageKHR(
                self.handle,
                std.math.maxInt(u64),
                self.next_image_acquired,
                .null_handle,
            );

            std.mem.swap(vk.Semaphore, &self.swap_images[result.image_index].image_acquired, &self.next_image_acquired);
            self.image_index = result.image_index;

            return switch (result.result) {
                .success => .optimal,
                .suboptimal_khr => .suboptimal,
                else => unreachable,
            };
        }

        const SwapImage = struct {
            image: vk.Image,
            view: vk.ImageView,
            image_acquired: vk.Semaphore,
            render_finished: vk.Semaphore,
            frame_fence: vk.Fence,

            fn init(device: *Engine.VulkanContext.Api.Device, image: vk.Image, format: vk.Format) !SwapImage {
                const view = try device.createImageView(&.{
                    .image = image,
                    .view_type = .@"2d",
                    .format = format,
                    .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
                    .subresource_range = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_mip_level = 0,
                        .level_count = 1,
                        .base_array_layer = 0,
                        .layer_count = 1,
                    },
                }, null);
                errdefer device.destroyImageView(view, null);

                const image_acquired = try device.createSemaphore(&.{}, null);
                errdefer device.destroySemaphore(image_acquired, null);

                const render_finished = try device.createSemaphore(&.{}, null);
                errdefer device.destroySemaphore(render_finished, null);

                const frame_fence = try device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
                errdefer device.destroyFence(frame_fence, null);

                return SwapImage{
                    .image = image,
                    .view = view,
                    .image_acquired = image_acquired,
                    .render_finished = render_finished,
                    .frame_fence = frame_fence,
                };
            }

            fn deinit(self: SwapImage, device: *Engine.VulkanContext.Api.Device) void {
                self.waitForFence(device) catch return;
                device.destroyImageView(self.view, null);
                device.destroySemaphore(self.image_acquired, null);
                device.destroySemaphore(self.render_finished, null);
                device.destroyFence(self.frame_fence, null);
            }

            fn waitForFence(self: SwapImage, device: *Engine.VulkanContext.Api.Device) !void {
                _ = try device.waitForFences(1, @ptrCast(&self.frame_fence), vk.TRUE, std.math.maxInt(u64));
            }
        };
    };
};

const Gui = struct {
    ctx: *c.ImGuiContext,

    const Device = Engine.VulkanContext.Api.Device;

    fn loader(name: [*c]const u8, instance: ?*anyopaque) callconv(.C) ?*const fn () callconv(.C) void {
        return c.glfwGetInstanceProcAddress(@ptrCast(instance), name);
    }

    fn init(window: *Engine.Window) !@This() {
        const ctx = c.ImGui_CreateContext(null) orelse return error.ErrorCreatingImguiContext;
        errdefer c.ImGui_DestroyContext(ctx);

        _ = c.cImGui_ImplVulkan_LoadFunctions(loader);

        const io = c.ImGui_GetIO();
        io.*.ConfigFlags |= c.ImGuiConfigFlags_NavEnableKeyboard;
        io.*.ConfigFlags |= c.ImGuiConfigFlags_NavEnableGamepad;

        _ = c.cImGui_ImplGlfw_InitForVulkan(window.handle, true);
        errdefer c.cImGui_ImplGlfw_Shutdown();

        return .{
            .ctx = ctx,
        };
    }

    fn deinit(self: *@This()) void {
        c.cImGui_ImplGlfw_Shutdown();
        c.ImGui_DestroyContext(self.ctx);
    }

    const GuiRenderer = struct {
        desc_pool: vk.DescriptorPool,
        pass: vk.RenderPass,

        cmd_pool: vk.CommandPool,
        cmd_bufs: []vk.CommandBuffer,

        fn init(engine: *Engine, swapchain: *Renderer.Swapchain) !@This() {
            const device = &engine.graphics.device;

            const cmd_pool = try device.createCommandPool(&.{
                .queue_family_index = engine.graphics.graphics_queue.family,
                .flags = .{
                    .reset_command_buffer_bit = true,
                },
            }, null);
            errdefer device.destroyCommandPool(cmd_pool, null);

            var desc_pool = try device.createDescriptorPool(&.{
                .flags = .{
                    .free_descriptor_set_bit = true,
                },
                .max_sets = 1,
                .pool_size_count = 1,
                .p_pool_sizes = &[_]vk.DescriptorPoolSize{.{
                    .type = .combined_image_sampler,
                    .descriptor_count = 1,
                }},
            }, null);
            errdefer device.destroyDescriptorPool(desc_pool, null);

            const color_attachment = vk.AttachmentDescription{
                .format = swapchain.surface_format.format,
                .samples = .{ .@"1_bit" = true },
                .load_op = .load,
                .store_op = .store,
                .stencil_load_op = .dont_care,
                .stencil_store_op = .dont_care,
                .initial_layout = .color_attachment_optimal,
                .final_layout = .present_src_khr,
            };

            const color_attachment_ref = vk.AttachmentReference{
                .attachment = 0,
                .layout = .color_attachment_optimal,
            };

            const subpass = vk.SubpassDescription{
                .pipeline_bind_point = .graphics,
                .color_attachment_count = 1,
                .p_color_attachments = @ptrCast(&color_attachment_ref),
            };
            const pass = try device.createRenderPass(&.{
                .attachment_count = 1,
                .p_attachments = @ptrCast(&color_attachment),
                .subpass_count = 1,
                .p_subpasses = @ptrCast(&subpass),
            }, null);
            errdefer device.destroyRenderPass(pass, null);

            const cmdbufs = try allocator.alloc(vk.CommandBuffer, swapchain.swap_images.len);
            errdefer allocator.free(cmdbufs);

            try device.allocateCommandBuffers(&.{
                .command_pool = cmd_pool,
                .level = .primary,
                .command_buffer_count = @intCast(cmdbufs.len),
            }, cmdbufs.ptr);
            errdefer device.freeCommandBuffers(cmd_pool, @intCast(cmdbufs.len), cmdbufs.ptr);

            var info = c.ImGui_ImplVulkan_InitInfo{
                .Instance = @as(*c.VkInstance, @ptrCast(&engine.graphics.instance.handle)).*,
                .PhysicalDevice = @as(*c.VkPhysicalDevice, @ptrCast(&engine.graphics.pdev)).*,
                .Device = @as(*c.VkDevice, @ptrCast(&engine.graphics.device.handle)).*,
                .QueueFamily = engine.graphics.graphics_queue.family,
                .Queue = @as(*c.VkQueue, @ptrCast(&engine.graphics.graphics_queue.handle)).*,
                .DescriptorPool = @as(*c.VkDescriptorPool, @ptrCast(&desc_pool)).*,
                .RenderPass = @as(*c.VkRenderPass, @ptrCast(@constCast(&pass))).*,
                .MinImageCount = 2,
                .ImageCount = @intCast(swapchain.swap_images.len),
                // .MSAASamples: VkSampleCountFlagBits = @import("std").mem.zeroes(VkSampleCountFlagBits),
                // .PipelineCache: VkPipelineCache = @import("std").mem.zeroes(VkPipelineCache),
                // .Subpass: u32 = @import("std").mem.zeroes(u32),
                // .UseDynamicRendering: bool = @import("std").mem.zeroes(bool),
                // .PipelineRenderingCreateInfo: VkPipelineRenderingCreateInfoKHR = @import("std").mem.zeroes(VkPipelineRenderingCreateInfoKHR),
                // .Allocator: [*c]const VkAllocationCallbacks = @import("std").mem.zeroes([*c]const VkAllocationCallbacks),
                // .CheckVkResultFn: ?*const fn (VkResult) callconv(.C) void = @import("std").mem.zeroes(?*const fn (VkResult) callconv(.C) void),
                // .MinAllocationSize: VkDeviceSize = @import("std").mem.zeroes(VkDeviceSize),
            };
            _ = c.cImGui_ImplVulkan_Init(&info);
            errdefer c.cImGui_ImplVulkan_Shutdown();

            return .{
                .desc_pool = desc_pool,
                .cmd_pool = cmd_pool,
                .pass = pass,
                .cmd_bufs = cmdbufs,
            };
        }

        fn render_start(_: *@This()) void {
            c.cImGui_ImplVulkan_NewFrame();
            c.cImGui_ImplGlfw_NewFrame();
            c.ImGui_NewFrame();
        }

        fn render_end(self: *@This(), device: *Device, renderer: *Renderer) !void {
            c.ImGui_Render();

            const draw_data = c.ImGui_GetDrawData();

            const index = renderer.swapchain.image_index;
            const cmdbuf = self.cmd_bufs[index];
            const framebuffer = renderer.framebuffers[index];

            try device.resetCommandBuffer(cmdbuf, .{ .release_resources_bit = true });
            try device.beginCommandBuffer(cmdbuf, &.{});

            device.cmdBeginRenderPass(cmdbuf, &.{
                .render_pass = self.pass,
                .framebuffer = framebuffer,
                .render_area = .{
                    .offset = .{ .x = 0, .y = 0 },
                    .extent = renderer.swapchain.extent,
                },
            }, .@"inline");

            c.cImGui_ImplVulkan_RenderDrawData(draw_data, @as(*c.VkCommandBuffer, @ptrCast(@constCast(&cmdbuf))).*);

            device.cmdEndRenderPass(cmdbuf);
            try device.endCommandBuffer(cmdbuf);
        }

        fn deinit(self: *@This(), device: *Device) void {
            defer device.destroyDescriptorPool(self.desc_pool, null);
            defer device.destroyRenderPass(self.pass, null);
            defer device.destroyCommandPool(self.cmd_pool, null);
            defer {
                device.freeCommandBuffers(self.cmd_pool, @intCast(self.cmd_bufs.len), self.cmd_bufs.ptr);
                allocator.free(self.cmd_bufs);
            }
            defer c.cImGui_ImplVulkan_Shutdown();
        }
    };
};

pub fn main() !void {
    {
        var engine = try Engine.init();
        defer engine.deinit();

        std.debug.print("using device: {s}\n", .{engine.graphics.props.device_name});

        var gui = try Gui.init(engine.window);
        defer gui.deinit();

        var renderer = try Renderer.init(&engine);
        defer renderer.deinit(&engine.graphics.device);

        var gui_renderer = try Gui.GuiRenderer.init(&engine, &renderer.swapchain);
        defer gui_renderer.deinit(&engine.graphics.device);

        var timer = try std.time.Timer.start();
        while (!engine.window.should_close()) {
            defer engine.window.tick();

            if (engine.window.is_minimized()) {
                continue;
            }

            gui_renderer.render_start();

            var show_demo_window: bool = true;
            c.ImGui_ShowDemoWindow(&show_demo_window);

            renderer.uniforms.tick(&timer, engine.window);

            try gui_renderer.render_end(&engine.graphics.device, &renderer);

            // multiple framebuffers => multiple descriptor sets => different buffers
            // big buffers that depends on the last frame's big buffer + multiple framebuffers => me sad
            // so just wait for one frame's queue to be empty before trying to render another frame
            try engine.graphics.device.queueWaitIdle(engine.graphics.graphics_queue.handle);

            const state = try renderer.present(&gui_renderer, &engine.graphics);
            // IDK: this never triggers :/
            if (state == .suboptimal) {
                std.debug.print("{any}\n", .{state});
            }

            if (engine.window.resize_fuse.unfuse() or state == .suboptimal) {
                renderer.deinit(&engine.graphics.device);
                renderer = try Renderer.init(&engine);

                gui_renderer.deinit(&engine.graphics.device);
                gui_renderer = try Gui.GuiRenderer.init(&engine, &renderer.swapchain);
            }
        }

        try renderer.swapchain.waitForAllFences(&engine.graphics.device);
        try engine.graphics.device.deviceWaitIdle();
    }

    // no defer cuz we don't want to print leaks when we error out
    _ = gpa.deinit();
}
