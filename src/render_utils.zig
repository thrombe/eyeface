const std = @import("std");

const vk = @import("vulkan");

const Engine = @import("engine.zig");

const main = @import("main.zig");
const allocator = main.allocator;

const Device = Engine.VulkanContext.Api.Device;

// TODO: don't depend on this
const Uniforms = @import("renderer.zig").Uniforms;

pub const Image = struct {
    image: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,

    pub const Args = struct {
        img_type: vk.ImageType,
        img_view_type: vk.ImageViewType,
        format: vk.Format,
        extent: vk.Extent3D,
        usage: vk.ImageUsageFlags = .{},
        view_aspect_mask: vk.ImageAspectFlags = .{},
    };

    pub fn new(ctx: *Engine.VulkanContext, v: Args) !@This() {
        const device = &ctx.device;

        const img = try device.createImage(&.{
            .image_type = v.img_type,
            .format = v.format,
            .extent = v.extent,
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = v.usage,
            .sharing_mode = .exclusive,
            .initial_layout = .undefined,
        }, null);
        errdefer device.destroyImage(img, null);

        const mem_reqs = device.getImageMemoryRequirements(img);
        const memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
        errdefer device.freeMemory(memory, null);
        try device.bindImageMemory(img, memory, 0);

        const view = try device.createImageView(&.{
            .image = img,
            .view_type = v.img_view_type,
            .format = v.format,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = v.view_aspect_mask,
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        errdefer device.destroyImageView(view, null);

        return .{
            .image = img,
            .memory = memory,
            .view = view,
        };
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        device.destroyImageView(self.view, null);
        device.freeMemory(self.memory, null);
        device.destroyImage(self.image, null);
    }
};

pub fn UniformBuffer(T: type) type {
    return struct {
        uniforms: T,
        buffer: vk.Buffer,
        memory: vk.DeviceMemory,
        dbi: vk.DescriptorBufferInfo,

        pub fn new(uniforms: T, ctx: *Engine.VulkanContext) !@This() {
            const device = &ctx.device;

            const buffer = try device.createBuffer(&.{
                .size = @sizeOf(T),
                .usage = .{
                    .uniform_buffer_bit = true,
                },
                .sharing_mode = .exclusive,
            }, null);
            errdefer device.destroyBuffer(buffer, null);
            const mem_req = device.getBufferMemoryRequirements(buffer);
            const memory = try ctx.allocate(mem_req, .{
                .host_visible_bit = true,
                .host_coherent_bit = true,
            });
            errdefer device.freeMemory(memory, null);
            try device.bindBufferMemory(buffer, memory, 0);

            return .{
                .uniforms = uniforms,
                .buffer = buffer,
                .memory = memory,
                .dbi = .{
                    .buffer = buffer,
                    .offset = 0,
                    .range = vk.WHOLE_SIZE,
                },
            };
        }

        pub fn deinit(self: *@This(), device: *Device) void {
            device.destroyBuffer(self.buffer, null);
            device.freeMemory(self.memory, null);
        }

        pub fn upload(self: *@This(), device: *Device) !void {
            const maybe_mapped = try device.mapMemory(self.memory, 0, @sizeOf(T), .{});
            const mapped = maybe_mapped orelse return error.MappingMemoryFailed;
            defer device.unmapMemory(self.memory);

            @memcpy(@as([*]u8, @ptrCast(mapped)), std.mem.asBytes(&self.uniforms));
        }

        pub fn layout_binding(_: *@This(), index: u32) vk.DescriptorSetLayoutBinding {
            return .{
                .binding = index,
                .descriptor_type = .uniform_buffer,
                .descriptor_count = 1,
                .stage_flags = .{
                    .vertex_bit = true,
                    .fragment_bit = true,
                    .compute_bit = true,
                },
            };
        }

        pub fn write_desc_set(self: *@This(), binding: u32, desc_set: vk.DescriptorSet) vk.WriteDescriptorSet {
            return .{
                .dst_set = desc_set,
                .dst_binding = binding,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .uniform_buffer,
                .p_buffer_info = @ptrCast(&self.dbi),
                // OOF: ??
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            };
        }
    };
}

pub const Buffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    dbi: vk.DescriptorBufferInfo,

    const Args = struct {
        size: u64,
        usage: vk.BufferUsageFlags = .{},
    };

    pub fn new_initialized(ctx: *Engine.VulkanContext, v: Args, val: anytype, pool: vk.CommandPool) !@This() {
        const this = try @This().new(
            ctx,
            .{ .size = v.size, .usage = v.usage.merge(.{
                .transfer_dst_bit = true,
            }) },
        );

        const staging_buffer = try ctx.device.createBuffer(&.{
            .size = v.size,
            .usage = .{ .transfer_src_bit = true },
            .sharing_mode = .exclusive,
        }, null);
        defer ctx.device.destroyBuffer(staging_buffer, null);
        const staging_mem_reqs = ctx.device.getBufferMemoryRequirements(staging_buffer);
        const staging_memory = try ctx.allocate(staging_mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
        defer ctx.device.freeMemory(staging_memory, null);
        try ctx.device.bindBufferMemory(staging_buffer, staging_memory, 0);

        {
            const data = try ctx.device.mapMemory(staging_memory, 0, vk.WHOLE_SIZE, .{});
            defer ctx.device.unmapMemory(staging_memory);

            const gpu_vertices: [*]@TypeOf(val) = @ptrCast(@alignCast(data));
            @memset(gpu_vertices[0 .. v.size / @as(u64, @sizeOf(@TypeOf(val)))], val);
        }

        try copyBuffer(ctx, &ctx.device, pool, this.buffer, staging_buffer, v.size);

        return this;
    }

    pub fn new(ctx: *Engine.VulkanContext, v: Args) !@This() {
        const device = &ctx.device;

        const buffer = try device.createBuffer(&.{
            .size = v.size,
            .usage = (vk.BufferUsageFlags{
                .storage_buffer_bit = true,
            }).merge(v.usage),
            .sharing_mode = .exclusive,
        }, null);
        errdefer device.destroyBuffer(buffer, null);

        const mem_reqs = device.getBufferMemoryRequirements(buffer);
        const memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
        errdefer device.freeMemory(memory, null);
        try device.bindBufferMemory(buffer, memory, 0);

        return .{
            .buffer = buffer,
            .memory = memory,
            .dbi = .{
                .buffer = buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            },
        };
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        device.destroyBuffer(self.buffer, null);
        device.freeMemory(self.memory, null);
    }

    pub fn layout_binding(_: *@This(), index: u32) vk.DescriptorSetLayoutBinding {
        return .{
            .binding = index,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        };
    }

    pub fn write_desc_set(self: *@This(), binding: u32, desc_set: vk.DescriptorSet) vk.WriteDescriptorSet {
        return .{
            .dst_set = desc_set,
            .dst_binding = binding,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = @ptrCast(&self.dbi),
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };
    }
};

pub const Swapchain = struct {
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

    pub fn present(self: *Swapchain, cmdbufs: []const vk.CommandBuffer, ctx: *Engine.VulkanContext, uniforms: *UniformBuffer(Uniforms)) !PresentState {
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

        try uniforms.upload(&ctx.device);

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

pub fn copyBuffer(
    ctx: *Engine.VulkanContext,
    device: *Device,
    pool: vk.CommandPool,
    dst: vk.Buffer,
    src: vk.Buffer,
    size: vk.DeviceSize,
) !void {
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