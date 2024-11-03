const std = @import("std");

const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", "1");
    // @cDefine("GLFW_INCLUDE_NONE", "1");
    @cInclude("GLFW/glfw3.h");
    @cInclude("GLFW/glfw3native.h");
});

const vk = @import("vulkan");

const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};
const apis: []const vk.ApiInfo = &.{
    .{
        .base_commands = .{
            .createInstance = true,
        },
        .instance_commands = .{
            .createDevice = true,
        },
    },
    vk.features.version_1_0,
    vk.features.version_1_1,
    vk.features.version_1_2,
    vk.features.version_1_3,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
};

const BaseDispatch = vk.BaseWrapper(apis);
const InstanceDispatch = vk.InstanceWrapper(apis);
const DeviceDispatch = vk.DeviceWrapper(apis);

const Instance = vk.InstanceProxy(apis);
const Device = vk.DeviceProxy(apis);

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

const SwapChain = struct {
    pub const Swapchain = struct {
        pub const PresentState = enum {
            optimal,
            suboptimal,
        };

        gc: *const GraphicsContext,
        surface_format: vk.SurfaceFormatKHR,
        present_mode: vk.PresentModeKHR,
        extent: vk.Extent2D,
        handle: vk.SwapchainKHR,

        swap_images: []SwapImage,
        image_index: u32,
        next_image_acquired: vk.Semaphore,

        pub fn init(gc: *const GraphicsContext, extent: vk.Extent2D) !Swapchain {
            return try initRecycle(gc, extent, .null_handle);
        }

        pub fn initRecycle(gc: *const GraphicsContext, extent: vk.Extent2D, old_handle: vk.SwapchainKHR) !Swapchain {
            const caps = try gc.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(gc.pdev, gc.surface);
            const actual_extent = findActualExtent(caps, extent);
            if (actual_extent.width == 0 or actual_extent.height == 0) {
                return error.InvalidSurfaceDimensions;
            }

            const surface_format = try findSurfaceFormat(gc);
            const present_mode = try findPresentMode(gc);

            var image_count = caps.min_image_count + 1;
            if (caps.max_image_count > 0) {
                image_count = @min(image_count, caps.max_image_count);
            }

            const qfi = [_]u32{ gc.graphics_queue.family, gc.present_queue.family };
            const sharing_mode: vk.SharingMode = if (gc.graphics_queue.family != gc.present_queue.family)
                .concurrent
            else
                .exclusive;

            const handle = try gc.device.createSwapchainKHR(&.{
                .surface = gc.surface,
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
            errdefer gc.device.destroySwapchainKHR(handle, null);

            if (old_handle != .null_handle) {
                // Apparently, the old swapchain handle still needs to be destroyed after recreating.
                gc.device.destroySwapchainKHR(old_handle, null);
            }

            const swap_images = try initSwapchainImages(gc, handle, surface_format.format);
            errdefer {
                for (swap_images) |si| si.deinit(gc);
                allocator.free(swap_images);
            }

            var next_image_acquired = try gc.device.createSemaphore(&.{}, null);
            errdefer gc.device.destroySemaphore(next_image_acquired, null);

            const result = try gc.device.acquireNextImageKHR(handle, std.math.maxInt(u64), next_image_acquired, .null_handle);
            if (result.result != .success) {
                return error.ImageAcquireFailed;
            }

            std.mem.swap(vk.Semaphore, &swap_images[result.image_index].image_acquired, &next_image_acquired);
            return Swapchain{
                .gc = gc,
                .surface_format = surface_format,
                .present_mode = present_mode,
                .extent = actual_extent,
                .handle = handle,
                .swap_images = swap_images,
                .image_index = result.image_index,
                .next_image_acquired = next_image_acquired,
            };
        }

        fn deinitExceptSwapchain(self: Swapchain) void {
            for (self.swap_images) |si| si.deinit(self.gc);
            allocator.free(self.swap_images);
            self.gc.device.destroySemaphore(self.next_image_acquired, null);
        }

        pub fn waitForAllFences(self: Swapchain) !void {
            for (self.swap_images) |si| si.waitForFence(self.gc) catch {};
        }

        pub fn deinit(self: Swapchain) void {
            self.deinitExceptSwapchain();
            self.gc.device.destroySwapchainKHR(self.handle, null);
        }

        pub fn recreate(self: *Swapchain, new_extent: vk.Extent2D) !void {
            const gc = self.gc;
            const old_handle = self.handle;
            self.deinitExceptSwapchain();
            self.* = try initRecycle(gc, new_extent, old_handle);
        }

        pub fn currentImage(self: Swapchain) vk.Image {
            return self.swap_images[self.image_index].image;
        }

        pub fn currentSwapImage(self: Swapchain) *const SwapImage {
            return &self.swap_images[self.image_index];
        }

        pub fn present(self: *Swapchain, cmdbuf: vk.CommandBuffer) !PresentState {
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
            try current.waitForFence(self.gc);
            try self.gc.device.resetFences(1, @ptrCast(&current.frame_fence));

            // Step 2: Submit the command buffer
            const wait_stage = [_]vk.PipelineStageFlags{.{ .top_of_pipe_bit = true }};
            try self.gc.device.queueSubmit(self.gc.graphics_queue.handle, 1, &[_]vk.SubmitInfo{.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&current.image_acquired),
                .p_wait_dst_stage_mask = &wait_stage,
                .command_buffer_count = 1,
                .p_command_buffers = @ptrCast(&cmdbuf),
                .signal_semaphore_count = 1,
                .p_signal_semaphores = @ptrCast(&current.render_finished),
            }}, current.frame_fence);

            // Step 3: Present the current frame
            _ = try self.gc.device.queuePresentKHR(self.gc.present_queue.handle, &.{
                .wait_semaphore_count = 1,
                .p_wait_semaphores = @ptrCast(&current.render_finished),
                .swapchain_count = 1,
                .p_swapchains = @ptrCast(&self.handle),
                .p_image_indices = @ptrCast(&self.image_index),
            });

            // Step 4: Acquire next frame
            const result = try self.gc.device.acquireNextImageKHR(
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
    };

    const SwapImage = struct {
        image: vk.Image,
        view: vk.ImageView,
        image_acquired: vk.Semaphore,
        render_finished: vk.Semaphore,
        frame_fence: vk.Fence,

        fn init(gc: *const GraphicsContext, image: vk.Image, format: vk.Format) !SwapImage {
            const view = try gc.device.createImageView(&.{
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
            errdefer gc.device.destroyImageView(view, null);

            const image_acquired = try gc.device.createSemaphore(&.{}, null);
            errdefer gc.device.destroySemaphore(image_acquired, null);

            const render_finished = try gc.device.createSemaphore(&.{}, null);
            errdefer gc.device.destroySemaphore(render_finished, null);

            const frame_fence = try gc.device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
            errdefer gc.device.destroyFence(frame_fence, null);

            return SwapImage{
                .image = image,
                .view = view,
                .image_acquired = image_acquired,
                .render_finished = render_finished,
                .frame_fence = frame_fence,
            };
        }

        fn deinit(self: SwapImage, gc: *const GraphicsContext) void {
            self.waitForFence(gc) catch return;
            gc.device.destroyImageView(self.view, null);
            gc.device.destroySemaphore(self.image_acquired, null);
            gc.device.destroySemaphore(self.render_finished, null);
            gc.device.destroyFence(self.frame_fence, null);
        }

        fn waitForFence(self: SwapImage, gc: *const GraphicsContext) !void {
            _ = try gc.device.waitForFences(1, @ptrCast(&self.frame_fence), vk.TRUE, std.math.maxInt(u64));
        }
    };

    fn initSwapchainImages(gc: *const GraphicsContext, swapchain: vk.SwapchainKHR, format: vk.Format) ![]SwapImage {
        const images = try gc.device.getSwapchainImagesAllocKHR(swapchain, allocator);
        defer allocator.free(images);

        const swap_images = try allocator.alloc(SwapImage, images.len);
        errdefer allocator.free(swap_images);

        var i: usize = 0;
        errdefer for (swap_images[0..i]) |si| si.deinit(gc);

        for (images) |image| {
            swap_images[i] = try SwapImage.init(gc, image, format);
            i += 1;
        }

        return swap_images;
    }

    fn findSurfaceFormat(gc: *const GraphicsContext) !vk.SurfaceFormatKHR {
        const preferred = vk.SurfaceFormatKHR{
            .format = .b8g8r8a8_srgb,
            .color_space = .srgb_nonlinear_khr,
        };

        const surface_formats = try gc.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(gc.pdev, gc.surface, allocator);
        defer allocator.free(surface_formats);

        for (surface_formats) |sfmt| {
            if (std.meta.eql(sfmt, preferred)) {
                return preferred;
            }
        }

        return surface_formats[0]; // There must always be at least one supported surface format
    }

    fn findPresentMode(gc: *const GraphicsContext) !vk.PresentModeKHR {
        const present_modes = try gc.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(gc.pdev, gc.surface, allocator);
        defer allocator.free(present_modes);

        const preferred = [_]vk.PresentModeKHR{
            .mailbox_khr,
            .immediate_khr,
        };

        for (preferred) |mode| {
            if (std.mem.indexOfScalar(vk.PresentModeKHR, present_modes, mode) != null) {
                return mode;
            }
        }

        return .fifo_khr;
    }

    fn findActualExtent(caps: vk.SurfaceCapabilitiesKHR, extent: vk.Extent2D) vk.Extent2D {
        if (caps.current_extent.width != 0xFFFF_FFFF) {
            return caps.current_extent;
        } else {
            return .{
                .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
                .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
            };
        }
    }
};
const GraphicsContext = struct {
    window: *c.GLFWwindow,
    vkb: BaseDispatch,
    instance: Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    mem_props: vk.PhysicalDeviceMemoryProperties,

    device: Device,
    graphics_queue: Queue,
    present_queue: Queue,

    pub const CommandBuffer = vk.CommandBufferProxy(apis);

    pub const Queue = struct {
        handle: vk.Queue,
        family: u32,

        fn init(device: Device, family: u32) Queue {
            return .{
                .handle = device.getDeviceQueue(family, 0),
                .family = family,
            };
        }
    };

    fn initWindow() !*c.GLFWwindow {
        _ = c.glfwSetErrorCallback(struct {
            fn callback(err: c_int, msg: [*c]const u8) callconv(.C) void {
                _ = err;
                std.debug.print("GLFW Error: {s}\n", .{msg});
            }
        }.callback);

        if (c.glfwInit() != c.GLFW_TRUE) return error.GlfwInitFailed;
        errdefer c.glfwTerminate();

        if (c.glfwPlatformSupported(c.GLFW_PLATFORM_WAYLAND) != c.GLFW_TRUE) {
            return error.WaylandNotSupported;
        }

        if (c.glfwPlatformSupported(c.GLFW_PLATFORM_X11) != c.GLFW_TRUE) {
            return error.X11NotSupported;
        }

        if (c.glfwVulkanSupported() != c.GLFW_TRUE) {
            return error.VulkanNotSupported;
        }

        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        const window = c.glfwCreateWindow(
            800,
            600,
            "yaaaaaaaaaa",
            null,
            null,
        ) orelse return error.WindowInitFailed;
        errdefer c.glfwDestroyWindow(window);

        return window;
    }

    fn initializeCandidate(instance: Instance, candidate: DeviceCandidate) !vk.Device {
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

        return try instance.createDevice(candidate.pdev, &.{
            .queue_create_info_count = queue_count,
            .p_queue_create_infos = &qci,
            .enabled_extension_count = required_device_extensions.len,
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
        }, null);
    }

    const DeviceCandidate = struct {
        pdev: vk.PhysicalDevice,
        props: vk.PhysicalDeviceProperties,
        queues: QueueAllocation,
    };

    const QueueAllocation = struct {
        graphics_family: u32,
        present_family: u32,
    };

    fn pickPhysicalDevice(
        instance: Instance,
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
        instance: Instance,
        pdev: vk.PhysicalDevice,
        surface: vk.SurfaceKHR,
    ) !?DeviceCandidate {
        if (!try checkExtensionSupport(instance, pdev)) {
            return null;
        }

        if (!try checkSurfaceSupport(instance, pdev, surface)) {
            return null;
        }

        if (try allocateQueues(instance, pdev, surface)) |allocation| {
            const props = instance.getPhysicalDeviceProperties(pdev);
            // if (props.device_type != .discrete_gpu) {
            //     return null;
            // }
            return DeviceCandidate{
                .pdev = pdev,
                .props = props,
                .queues = allocation,
            };
        }

        return null;
    }

    fn allocateQueues(instance: Instance, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !?QueueAllocation {
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

        if (graphics_family != null and present_family != null) {
            return QueueAllocation{
                .graphics_family = graphics_family.?,
                .present_family = present_family.?,
            };
        }

        return null;
    }

    fn checkSurfaceSupport(instance: Instance, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
        var format_count: u32 = undefined;
        _ = try instance.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);

        var present_mode_count: u32 = undefined;
        _ = try instance.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);

        return format_count > 0 and present_mode_count > 0;
    }

    fn checkExtensionSupport(
        instance: Instance,
        pdev: vk.PhysicalDevice,
    ) !bool {
        const propsv = try instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, allocator);
        defer allocator.free(propsv);

        for (required_device_extensions) |ext| {
            for (propsv) |props| {
                if (std.mem.eql(u8, std.mem.span(ext), std.mem.sliceTo(&props.extension_name, 0))) {
                    break;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    pub fn findMemoryTypeIndex(self: GraphicsContext, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
        for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
            if (memory_type_bits & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(flags)) {
                return @truncate(i);
            }
        }

        return error.NoSuitableMemoryType;
    }

    pub fn allocate(self: GraphicsContext, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
        return try self.device.allocateMemory(&.{
            .allocation_size = requirements.size,
            .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
        }, null);
    }

    fn init() !@This() {
        const window = try initWindow();
        errdefer c.glfwDestroyWindow(window);

        const vkb = try BaseDispatch.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(&c.glfwGetInstanceProcAddress)));

        var glfw_exts_count: u32 = 0;
        const glfw_exts = c.glfwGetRequiredInstanceExtensions(&glfw_exts_count);
        const instance = vkb.createInstance(&.{
            .p_application_info = &.{
                .p_application_name = "yaaaaaaaaaaaaaaa",
                .api_version = vk.API_VERSION_1_3,
                .application_version = vk.makeApiVersion(0, 0, 1, 0),
                .engine_version = vk.makeApiVersion(0, 0, 1, 0),
            },
            .enabled_extension_count = glfw_exts_count,
            .pp_enabled_extension_names = @ptrCast(glfw_exts),
        }, null) catch |e| {
            std.debug.print("{any}\n", .{e});
            return e;
        };

        const vki = try allocator.create(InstanceDispatch);
        errdefer allocator.destroy(vki);
        vki.* = try InstanceDispatch.load(instance, vkb.dispatch.vkGetInstanceProcAddr);
        const vkinstance: Instance = Instance.init(instance, vki);
        errdefer vkinstance.destroyInstance(null);

        var surface: vk.SurfaceKHR = undefined;
        if (c.glfwCreateWindowSurface(@as(*const c.VkInstance, @ptrCast(&vkinstance.handle)).*, window, null, @ptrCast(&surface)) != c.VK_SUCCESS) {
            return error.SurfaceInitFailed;
        }

        const candidate = try pickPhysicalDevice(vkinstance, surface);
        const pdev = candidate.pdev;
        const props = candidate.props;

        const dev = try initializeCandidate(vkinstance, candidate);

        const vkd = try allocator.create(DeviceDispatch);
        errdefer allocator.destroy(vkd);
        vkd.* = try DeviceDispatch.load(dev, vkinstance.wrapper.dispatch.vkGetDeviceProcAddr);
        const device = Device.init(dev, vkd);
        errdefer device.destroyDevice(null);

        const graphics_queue = Queue.init(device, candidate.queues.graphics_family);
        const present_queue = Queue.init(device, candidate.queues.present_family);

        const mem_props = vkinstance.getPhysicalDeviceMemoryProperties(pdev);

        return .{
            .window = window,
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
        c.glfwDestroyWindow(self.window);
        c.glfwTerminate();

        // Don't forget to free the tables to prevent a memory leak.
        allocator.destroy(self.device.wrapper);
        allocator.destroy(self.instance.wrapper);
    }
};

pub fn main() !void {
    var gc = try GraphicsContext.init();
    defer gc.deinit();

    std.debug.print("using device: {s}\n", .{gc.props.device_name});

    var extent = vk.Extent2D{ .width = 800, .height = 600 };

    var swapchain = try SwapChain.Swapchain.init(&gc, .{
        .width = 800,
        .height = 600,
    });
    defer swapchain.deinit();

    const pipeline_layout = try gc.device.createPipelineLayout(&.{
        .flags = .{},
        .set_layout_count = 0,
        .p_set_layouts = undefined,
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    defer gc.device.destroyPipelineLayout(pipeline_layout, null);

    const render_pass = try createRenderPass(&gc, swapchain);
    defer gc.device.destroyRenderPass(render_pass, null);

    const pipeline = try createPipeline(&gc, pipeline_layout, render_pass);
    defer gc.device.destroyPipeline(pipeline, null);

    var framebuffers = try createFramebuffers(&gc, render_pass, swapchain);
    defer destroyFramebuffers(&gc, framebuffers);

    const pool = try gc.device.createCommandPool(&.{
        .queue_family_index = gc.graphics_queue.family,
    }, null);
    defer gc.device.destroyCommandPool(pool, null);

    const buffer = try gc.device.createBuffer(&.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer gc.device.destroyBuffer(buffer, null);
    const mem_reqs = gc.device.getBufferMemoryRequirements(buffer);
    const memory = try gc.allocate(mem_reqs, .{ .device_local_bit = true });
    defer gc.device.freeMemory(memory, null);
    try gc.device.bindBufferMemory(buffer, memory, 0);

    try uploadVertices(&gc, pool, buffer);

    var cmdbufs = try createCommandBuffers(
        &gc,
        pool,
        buffer,
        swapchain.extent,
        render_pass,
        pipeline,
        framebuffers,
    );
    defer destroyCommandBuffers(&gc, pool, cmdbufs);

    while (c.glfwWindowShouldClose(gc.window) == c.GLFW_FALSE) {
        var w: c_int = undefined;
        var h: c_int = undefined;
        c.glfwGetFramebufferSize(gc.window, &w, &h);

        // Don't present or resize swapchain while the window is minimized
        if (w == 0 or h == 0) {
            c.glfwPollEvents();
            continue;
        }

        const cmdbuf = cmdbufs[swapchain.image_index];

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => SwapChain.Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal or extent.width != @as(u32, @intCast(w)) or extent.height != @as(u32, @intCast(h))) {
            extent.width = @intCast(w);
            extent.height = @intCast(h);
            try swapchain.recreate(extent);

            destroyFramebuffers(&gc, framebuffers);
            framebuffers = try createFramebuffers(&gc, render_pass, swapchain);

            destroyCommandBuffers(&gc, pool, cmdbufs);
            cmdbufs = try createCommandBuffers(
                &gc,
                pool,
                buffer,
                swapchain.extent,
                render_pass,
                pipeline,
                framebuffers,
            );
        }

        c.glfwPollEvents();
    }

    try swapchain.waitForAllFences();
    try gc.device.deviceWaitIdle();
}

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
const vertices = [_]Vertex{
    .{ .pos = .{ 0, -0.5 }, .color = .{ 1, 0, 0 } },
    .{ .pos = .{ 0.5, 0.5 }, .color = .{ 0, 1, 0 } },
    .{ .pos = .{ -0.5, 0.5 }, .color = .{ 0, 0, 1 } },
};
fn uploadVertices(gc: *const GraphicsContext, pool: vk.CommandPool, buffer: vk.Buffer) !void {
    const staging_buffer = try gc.device.createBuffer(&.{
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
    }, null);
    defer gc.device.destroyBuffer(staging_buffer, null);
    const mem_reqs = gc.device.getBufferMemoryRequirements(staging_buffer);
    const staging_memory = try gc.allocate(mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer gc.device.freeMemory(staging_memory, null);
    try gc.device.bindBufferMemory(staging_buffer, staging_memory, 0);

    {
        const data = try gc.device.mapMemory(staging_memory, 0, vk.WHOLE_SIZE, .{});
        defer gc.device.unmapMemory(staging_memory);

        const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
        @memcpy(gpu_vertices, vertices[0..]);
    }

    try copyBuffer(gc, pool, buffer, staging_buffer, @sizeOf(@TypeOf(vertices)));
}

fn copyBuffer(gc: *const GraphicsContext, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    var cmdbuf_handle: vk.CommandBuffer = undefined;
    try gc.device.allocateCommandBuffers(&.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf_handle));
    defer gc.device.freeCommandBuffers(pool, 1, @ptrCast(&cmdbuf_handle));

    const cmdbuf = GraphicsContext.CommandBuffer.init(cmdbuf_handle, gc.device.wrapper);

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
    try gc.device.queueSubmit(gc.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
    try gc.device.queueWaitIdle(gc.graphics_queue.handle);
}

fn createCommandBuffers(
    gc: *const GraphicsContext,
    pool: vk.CommandPool,
    buffer: vk.Buffer,
    extent: vk.Extent2D,
    render_pass: vk.RenderPass,
    pipeline: vk.Pipeline,
    framebuffers: []vk.Framebuffer,
) ![]vk.CommandBuffer {
    const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
    errdefer allocator.free(cmdbufs);

    try gc.device.allocateCommandBuffers(&.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = @intCast(cmdbufs.len),
    }, cmdbufs.ptr);
    errdefer gc.device.freeCommandBuffers(pool, @intCast(cmdbufs.len), cmdbufs.ptr);

    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @floatFromInt(extent.width),
        .height = @floatFromInt(extent.height),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    for (cmdbufs, framebuffers) |cmdbuf, framebuffer| {
        try gc.device.beginCommandBuffer(cmdbuf, &.{});

        gc.device.cmdSetViewport(cmdbuf, 0, 1, @ptrCast(&viewport));
        gc.device.cmdSetScissor(cmdbuf, 0, 1, @ptrCast(&scissor));

        // This needs to be a separate definition - see https://github.com/ziglang/zig/issues/7627.
        const render_area = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = extent,
        };

        gc.device.cmdBeginRenderPass(cmdbuf, &.{
            .render_pass = render_pass,
            .framebuffer = framebuffer,
            .render_area = render_area,
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear),
        }, .@"inline");

        gc.device.cmdBindPipeline(cmdbuf, .graphics, pipeline);
        const offset = [_]vk.DeviceSize{0};
        gc.device.cmdBindVertexBuffers(cmdbuf, 0, 1, @ptrCast(&buffer), &offset);
        gc.device.cmdDraw(cmdbuf, vertices.len, 1, 0, 0);

        gc.device.cmdEndRenderPass(cmdbuf);
        try gc.device.endCommandBuffer(cmdbuf);
    }

    return cmdbufs;
}

fn destroyCommandBuffers(gc: *const GraphicsContext, pool: vk.CommandPool, cmdbufs: []vk.CommandBuffer) void {
    gc.device.freeCommandBuffers(pool, @truncate(cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

fn createFramebuffers(gc: *const GraphicsContext, render_pass: vk.RenderPass, swapchain: SwapChain.Swapchain) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.device.destroyFramebuffer(fb, null);

    for (framebuffers) |*fb| {
        fb.* = try gc.device.createFramebuffer(&.{
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swapchain.swap_images[i].view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }

    return framebuffers;
}

fn destroyFramebuffers(gc: *const GraphicsContext, framebuffers: []const vk.Framebuffer) void {
    for (framebuffers) |fb| gc.device.destroyFramebuffer(fb, null);
    allocator.free(framebuffers);
}

fn createRenderPass(gc: *const GraphicsContext, swapchain: SwapChain.Swapchain) !vk.RenderPass {
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

    return try gc.device.createRenderPass(&.{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
    }, null);
}

const vert_spv align(@alignOf(u32)) = @embedFile("vertex_shader").*;
const frag_spv align(@alignOf(u32)) = @embedFile("fragment_shader").*;
fn createPipeline(
    gc: *const GraphicsContext,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
) !vk.Pipeline {
    const vert = try gc.device.createShaderModule(&.{
        .code_size = vert_spv.len,
        .p_code = @ptrCast(&vert_spv),
    }, null);
    defer gc.device.destroyShaderModule(vert, null);

    const frag = try gc.device.createShaderModule(&.{
        .code_size = frag_spv.len,
        .p_code = @ptrCast(&frag_spv),
    }, null);
    defer gc.device.destroyShaderModule(frag, null);

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
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = true },
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
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try gc.device.createGraphicsPipelines(
        .null_handle,
        1,
        @ptrCast(&gpci),
        null,
        @ptrCast(&pipeline),
    );
    return pipeline;
}
