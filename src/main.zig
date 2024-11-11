const std = @import("std");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "1");
    // @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("GLFW/glfw3.h");
    @cInclude("GLFW/glfw3native.h");
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
    pass: vk.RenderPass,
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
    const Vertex = struct {
        const binding_description = vk.VertexInputBindingDescription{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .input_rate = .vertex,
        };
        const attribute_description = [_]vk.VertexInputAttributeDescription{
            .{
                .binding = 0,
                .location = 0,
                .format = .r32g32_sfloat,
                .offset = @offsetOf(Vertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = .r32g32b32_sfloat,
                .offset = @offsetOf(Vertex, "color"),
            },
        };

        pos: [2]f32,
        color: [3]f32,
    };

    fn init(engine: *Engine) !@This() {
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

        var ctx = &engine.graphics;
        const device = &ctx.device;
        var swapchain = try Swapchain.init(ctx, engine.window.extent);
        errdefer swapchain.deinit(device);

        const pass = blk: {
            const color_attachment = vk.AttachmentDescription{
                .format = swapchain.surface_format.format,
                .samples = .{ .@"1_bit" = true },
                .load_op = .clear,
                .store_op = .store,
                .stencil_load_op = .dont_care,
                .stencil_store_op = .dont_care,
                .initial_layout = .undefined,
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

            break :blk try device.createRenderPass(&.{
                .attachment_count = 1,
                .p_attachments = @ptrCast(&color_attachment),
                .subpass_count = 1,
                .p_subpasses = @ptrCast(&subpass),
            }, null);
        };
        errdefer device.destroyRenderPass(pass, null);

        const pipeline_layout = try device.createPipelineLayout(&.{
            .flags = .{},
            .set_layout_count = 0,
            .p_set_layouts = undefined,
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

            var i: usize = 0;
            errdefer for (framebuffers[0..i]) |fb| device.destroyFramebuffer(fb, null);
            for (framebuffers) |*fb| {
                fb.* = try device.createFramebuffer(&.{
                    .render_pass = pass,
                    .attachment_count = 1,
                    .p_attachments = @ptrCast(&swapchain.swap_images[i].view),
                    .width = swapchain.extent.width,
                    .height = swapchain.extent.height,
                    .layers = 1,
                }, null);
                i += 1;
            }

            break :blk framebuffers;
        };
        errdefer {
            for (framebuffers) |fb| device.destroyFramebuffer(fb, null);
            allocator.free(framebuffers);
        }

        const pool = try device.createCommandPool(&.{
            .queue_family_index = ctx.graphics_queue.family,
        }, null);
        errdefer device.destroyCommandPool(pool, null);

        var vertices = std.ArrayList(Vertex).init(allocator);
        defer vertices.deinit();
        var rng = std.Random.DefaultPrng.init(0);
        const p1 = utils.Vec4{ .x = 0, .y = -0.5 };
        const p2 = utils.Vec4{ .x = 0.5, .y = 0.5 };
        const p3 = utils.Vec4{ .x = -0.5, .y = 0.5 };
        for (0..10000) |_| {
            var pos = utils.Vec4{};
            for (0..10) |_| {
                const p = switch (rng.next() % 3) {
                    0 => p1,
                    1 => p2,
                    2 => p3,
                    else => unreachable,
                };
                pos.x += p.x;
                pos.x /= 2.0;
                pos.y += p.y;
                pos.y /= 2.0;
            }
            try vertices.append(.{ .pos = .{ pos.x, pos.y }, .color = .{ 1, 1, 1 } });
        }
        // try vertices.append(.{ .pos = .{ 0, -0.5 }, .color = .{ 1, 1, 1 } });
        // try vertices.append(.{ .pos = .{ 0.5, 0.5 }, .color = .{ 1, 1, 1 } });
        // try vertices.append(.{ .pos = .{ -0.5, 0.5 }, .color = .{ 1, 1, 1 } });
        const vertex_buffer = blk: {
            const buffer = try device.createBuffer(&.{
                .size = @sizeOf(Vertex) * vertices.items.len,
                .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
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
                const offset = [_]vk.DeviceSize{0};
                device.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&vertex_buffer.buffer), &offset);
                device.cmdDraw(cmdbuf, @intCast(vertices.items.len), 1, 0, 0);

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
            .pass = pass,
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
        device.freeCommandBuffers(self.command_pool, @truncate(self.command_buffers.len), self.command_buffers.ptr);
        allocator.free(self.command_buffers);

        device.destroyBuffer(self.vertex_buffer, null);
        device.freeMemory(self.vertex_buffer_memory, null);
        device.destroyCommandPool(self.command_pool, null);

        for (self.framebuffers) |fb| device.destroyFramebuffer(fb, null);
        allocator.free(self.framebuffers);

        device.destroyPipeline(self.pipeline, null);
        device.destroyPipelineLayout(self.pipeline_layout, null);
        device.destroyRenderPass(self.pass, null);
        self.swapchain.deinit(device);
    }

    fn present(self: *@This(), ctx: *Engine.VulkanContext) !Swapchain.PresentState {
        const cmdbuf = self.command_buffers[self.swapchain.image_index];

        return self.swapchain.present(cmdbuf, ctx) catch |err| switch (err) {
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

        pub fn present(self: *Swapchain, cmdbuf: vk.CommandBuffer, ctx: *Engine.VulkanContext) !PresentState {
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

            // Step 2: Submit the command buffer
            const wait_stage = [_]vk.PipelineStageFlags{.{ .top_of_pipe_bit = true }};
            try ctx.device.queueSubmit(ctx.graphics_queue.handle, 1, &[_]vk.SubmitInfo{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&current.image_acquired),
                .p_wait_dst_stage_mask = &wait_stage,
                .command_buffer_count = 1,
                .p_command_buffers = @ptrCast(&cmdbuf),
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

pub fn main() !void {
    {
        var engine = try Engine.init();
        defer engine.deinit();

        std.debug.print("using device: {s}\n", .{engine.graphics.props.device_name});

        var renderer = try Renderer.init(&engine);
        defer renderer.deinit(&engine.graphics.device);

        while (!engine.window.should_close()) {
            defer engine.window.tick();

            if (engine.window.is_minimized()) {
                continue;
            }

            const state = try renderer.present(&engine.graphics);
            // IDK: this never triggers :/
            if (state == .suboptimal) {
                std.debug.print("{any}\n", .{state});
            }

            if (engine.window.resize_fuse.unfuse() or state == .suboptimal) {
                renderer.deinit(&engine.graphics.device);
                renderer = try Renderer.init(&engine);
            }
        }

        try renderer.swapchain.waitForAllFences(&engine.graphics.device);
        try engine.graphics.device.deviceWaitIdle();
    }

    // no defer cuz we don't want to print leaks when we error out
    _ = gpa.deinit();
}
