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

const state = @import("state.zig");
const AppState = state.AppState;

const main = @import("main.zig");
const allocator = main.allocator;

const Renderer = @This();

swapchain: Swapchain,
uniforms: Uniforms,
uniform_buffer: vk.Buffer,
uniform_memory: vk.DeviceMemory,
vertex_buffer_memory: vk.DeviceMemory,
vertex_buffer: vk.Buffer,
// - [Depth buffering - Vulkan Tutorial](https://vulkan-tutorial.com/Depth_buffering)
// - [Setting up depth buffer - Vulkan Guide](https://vkguide.dev/docs/chapter-3/depth_buffer/)
depth_image: vk.Image,
depth_buffer_memory: vk.DeviceMemory,
depth_image_view: vk.ImageView,
voxel_buffer: vk.Buffer,
voxel_buffer_memory: vk.DeviceMemory,
occlusion_buffer: vk.Buffer,
occlusion_buffer_memory: vk.DeviceMemory,
screen_buffer: vk.Buffer,
screen_buffer_memory: vk.DeviceMemory,
screen_depth_buffer: vk.Buffer,
screen_depth_buffer_memory: vk.DeviceMemory,
descriptor_pool: vk.DescriptorPool,
frag_descriptor_set_layout: vk.DescriptorSetLayout,
frag_descriptor_set: vk.DescriptorSet,
compute_descriptor_set_layout: vk.DescriptorSetLayout,
compute_descriptor_set: vk.DescriptorSet,
pass: vk.RenderPass,
compute_pipeline_layout: vk.PipelineLayout,
compute_pipelines: []vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
pipeline: vk.Pipeline,
// framebuffers are objects containing views of swapchain images
framebuffers: []vk.Framebuffer,
command_pool: vk.CommandPool,
// command buffers are recordings of a bunch of commands that a gpu can execute
command_buffers: []vk.CommandBuffer,

const Device = Engine.VulkanContext.Api.Device;
pub const Vertex = extern struct {
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

pub const Uniforms = extern struct {
    transforms: TransformSet,
    world_to_screen: Mat4x4,
    eye: Vec4,
    mouse: extern struct { x: i32, y: i32, left: u32, right: u32 },
    occlusion_color: Vec4,
    sparse_color: Vec4,
    occlusion_multiplier: f32,
    occlusion_attenuation: f32,
    iterations: u32,
    voxelization_iterations: u32,
    voxel_grid_center: Vec4,
    voxel_grid_half_size: f32,
    voxel_grid_side: u32,
    frame: u32,
    time: f32,
    width: u32,
    height: u32,

    pub const TransformSet = transform.TransformSet(5);
};

pub fn init(engine: *Engine, app_state: *AppState) !@This() {
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
    try vertices.appendNTimes(.{ .pos = .{ 0, 0, 0, 1 } }, 64 * app_state.points_x_64);

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

    // apparently you don't need to create more than 1 depth buffer even if you have many
    // framebuffers
    const depth_image = blk: {
        const img = try device.createImage(&.{
            .image_type = .@"2d",
            .format = .d32_sfloat,
            .extent = .{
                .width = engine.window.extent.width,
                .height = engine.window.extent.height,
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .optimal,
            .usage = .{
                .depth_stencil_attachment_bit = true,
            },
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
            .view_type = .@"2d",
            .format = .d32_sfloat,
            .components = .{
                .r = .identity,
                .g = .identity,
                .b = .identity,
                .a = .identity,
            },
            .subresource_range = .{
                .aspect_mask = .{ .depth_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        errdefer device.destroyImageView(view, null);

        break :blk .{
            .image = img,
            .memory = memory,
            .view = view,
        };
    };
    errdefer {
        device.destroyImageView(depth_image.view, null);
        device.freeMemory(depth_image.memory, null);
        device.destroyImage(depth_image.image, null);
    }

    const voxels = blk: {
        const vol_size = app_state.voxels.side;
        const voxel_buffer = try device.createBuffer(&.{
            .size = @sizeOf(u32) * vol_size * vol_size * vol_size,
            .usage = .{
                .storage_buffer_bit = true,
            },
            .sharing_mode = .exclusive,
        }, null);
        errdefer device.destroyBuffer(voxel_buffer, null);

        const mem_reqs = device.getBufferMemoryRequirements(voxel_buffer);
        const voxel_memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
        errdefer device.freeMemory(voxel_memory, null);
        try device.bindBufferMemory(voxel_buffer, voxel_memory, 0);

        const occlusion_buffer = try device.createBuffer(&.{
            .size = @sizeOf(u32) * vol_size * vol_size * vol_size,
            .usage = .{
                .storage_buffer_bit = true,
            },
            .sharing_mode = .exclusive,
        }, null);
        errdefer device.destroyBuffer(occlusion_buffer, null);

        const occlusion_memory = try ctx.allocate(
            device.getBufferMemoryRequirements(occlusion_buffer),
            .{ .device_local_bit = true },
        );
        errdefer device.freeMemory(occlusion_memory, null);
        try device.bindBufferMemory(occlusion_buffer, occlusion_memory, 0);

        break :blk .{
            .voxel_buffer_memory = voxel_memory,
            .voxel_buffer = voxel_buffer,
            .occlusion_buffer_memory = occlusion_memory,
            .occlusion_buffer = occlusion_buffer,
        };
    };
    errdefer {
        device.destroyBuffer(voxels.voxel_buffer, null);
        device.freeMemory(voxels.voxel_buffer_memory, null);
        device.destroyBuffer(voxels.occlusion_buffer, null);
        device.freeMemory(voxels.occlusion_buffer_memory, null);
    }

    const screen = blk: {
        const screen_buffer = try device.createBuffer(&.{
            .size = @sizeOf(f32) * 4 * engine.window.extent.width * engine.window.extent.height,
            .usage = .{
                .storage_buffer_bit = true,
            },
            .sharing_mode = .exclusive,
        }, null);
        errdefer device.destroyBuffer(screen_buffer, null);

        const mem_reqs = device.getBufferMemoryRequirements(screen_buffer);
        const screen_memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
        errdefer device.freeMemory(screen_memory, null);
        try device.bindBufferMemory(screen_buffer, screen_memory, 0);

        break :blk .{ .buffer = screen_buffer, .memory = screen_memory };
    };
    errdefer {
        device.destroyBuffer(screen.buffer, null);
        device.freeMemory(screen.memory, null);
    }

    const screen_depth = blk: {
        const screen_buffer = try device.createBuffer(&.{
            .size = @sizeOf(f32) * engine.window.extent.width * engine.window.extent.height,
            .usage = .{
                .storage_buffer_bit = true,
            },
            .sharing_mode = .exclusive,
        }, null);
        errdefer device.destroyBuffer(screen_buffer, null);

        const mem_reqs = device.getBufferMemoryRequirements(screen_buffer);
        const screen_memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
        errdefer device.freeMemory(screen_memory, null);
        try device.bindBufferMemory(screen_buffer, screen_memory, 0);

        break :blk .{ .buffer = screen_buffer, .memory = screen_memory };
    };
    errdefer {
        device.destroyBuffer(screen_depth.buffer, null);
        device.freeMemory(screen_depth.memory, null);
    }

    const frag_bindings = [_]vk.DescriptorSetLayoutBinding{
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
        .{
            .binding = 2,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
        .{
            .binding = 3,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
        .{
            .binding = 4,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
    };
    const frag_desc_set_layout = try device.createDescriptorSetLayout(&.{
        .flags = .{},
        .binding_count = frag_bindings.len,
        .p_bindings = &frag_bindings,
    }, null);
    errdefer device.destroyDescriptorSetLayout(frag_desc_set_layout, null);

    const compute_bindings = [_]vk.DescriptorSetLayoutBinding{
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
        .{
            .binding = 2,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
        .{
            .binding = 3,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
        .{
            .binding = 4,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
        .{
            .binding = 5,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{
                .vertex_bit = true,
                .fragment_bit = true,
                .compute_bit = true,
            },
        },
    };
    const compute_desc_set_layout = try device.createDescriptorSetLayout(&.{
        .flags = .{},
        .binding_count = compute_bindings.len,
        .p_bindings = &compute_bindings,
    }, null);
    errdefer device.destroyDescriptorSetLayout(compute_desc_set_layout, null);
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{
            .type = .uniform_buffer,
            .descriptor_count = 1,
        },
        .{
            .type = .storage_buffer,
            .descriptor_count = 5,
        },
    };
    const desc_pool = try device.createDescriptorPool(&.{
        .max_sets = 2,
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
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
    const frag_desc_set_updates = [_]vk.WriteDescriptorSet{
        .{
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
        },
        .{
            .dst_set = frag_desc_set,
            .dst_binding = 1,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = voxels.voxel_buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = frag_desc_set,
            .dst_binding = 2,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = voxels.occlusion_buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = frag_desc_set,
            .dst_binding = 3,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = screen.buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = frag_desc_set,
            .dst_binding = 4,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = screen_depth.buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    };
    device.updateDescriptorSets(frag_desc_set_updates.len, &frag_desc_set_updates, 0, null);
    const compute_desc_set_updates = [_]vk.WriteDescriptorSet{
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
        .{
            .dst_set = compute_desc_set,
            .dst_binding = 2,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = voxels.voxel_buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = compute_desc_set,
            .dst_binding = 3,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = voxels.occlusion_buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = compute_desc_set,
            .dst_binding = 4,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = screen.buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        .{
            .dst_set = compute_desc_set,
            .dst_binding = 5,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = &[_]vk.DescriptorBufferInfo{.{
                .buffer = screen_depth.buffer,
                .offset = 0,
                .range = vk.WHOLE_SIZE,
            }},
            // OOF: ??
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    };
    device.updateDescriptorSets(compute_desc_set_updates.len, &compute_desc_set_updates, 0, null);

    var compiler = utils.Glslc.Compiler{ .opt = .fast, .env = .vulkan1_3 };
    const vert_spv = blk: {
        compiler.stage = .vertex;
        const vert: utils.Glslc.Compiler.Code = .{ .path = .{
            .main = "./src/shader.glsl",
            .include = &[_][]const u8{},
            .definitions = &[_][]const u8{ "EYEFACE_RENDER", "EYEFACE_VERT" },
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
            .main = "./src/shader.glsl",
            .include = &[_][]const u8{},
            .definitions = &[_][]const u8{ "EYEFACE_RENDER", "EYEFACE_FRAG" },
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

    const pass = blk: {
        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = @ptrCast(&vk.AttachmentReference{
                .attachment = 0,
                .layout = .color_attachment_optimal,
            }),
            .p_depth_stencil_attachment = @ptrCast(&vk.AttachmentReference{
                .attachment = 1,
                .layout = .depth_stencil_attachment_optimal,
            }),
        };

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

        const depth_attachment = vk.AttachmentDescription{
            .format = .d32_sfloat,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .dont_care,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .depth_stencil_attachment_optimal,
        };

        const attachments = [_]vk.AttachmentDescription{ color_attachment, depth_attachment };

        const deps = [_]vk.SubpassDependency{
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{
                    .color_attachment_output_bit = true,
                },
                .dst_stage_mask = .{
                    .color_attachment_output_bit = true,
                },
                // src_access_mask: AccessFlags = .{},
                .dst_access_mask = .{
                    .color_attachment_write_bit = true,
                },
                // dependency_flags: DependencyFlags = .{},
            },
            .{
                .src_subpass = vk.SUBPASS_EXTERNAL,
                .dst_subpass = 0,
                .src_stage_mask = .{
                    .early_fragment_tests_bit = true,
                },
                .dst_stage_mask = .{
                    .early_fragment_tests_bit = true,
                },
                // src_access_mask: AccessFlags = .{},
                .dst_access_mask = .{
                    .depth_stencil_attachment_write_bit = true,
                },
                // dependency_flags: DependencyFlags = .{},
            },
        };

        break :blk try device.createRenderPass(&.{
            .attachment_count = @intCast(attachments.len),
            .p_attachments = &attachments,
            .subpass_count = 1,
            .p_subpasses = @ptrCast(&subpass),
            .dependency_count = @intCast(deps.len),
            .p_dependencies = &deps,
        }, null);
    };
    errdefer device.destroyRenderPass(pass, null);

    const compute_pipeline_layout = try device.createPipelineLayout(&.{
        // .flags: PipelineLayoutCreateFlags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&compute_desc_set_layout),
    }, null);
    errdefer device.destroyPipelineLayout(compute_pipeline_layout, null);
    const compute_pipelines = blk: {
        var pipelines = [_]struct {
            path: [:0]const u8,
            define: []const u8,
            group_x: u32 = 1,
            group_y: u32 = 1,
            group_z: u32 = 1,
            pipeline: vk.Pipeline = undefined,
        }{
            .{
                .path = "./src/shader.glsl",
                .define = "EYEFACE_CLEAR_BUFS",
                .group_x = app_state.voxels.side * app_state.voxels.side * app_state.voxels.side / 64,
            },
            .{
                .path = "./src/shader.glsl",
                .define = "EYEFACE_VOXELIZE",
                // .group_x = 10000 * 64 / 64,
                .group_x = @intCast(vertices.items.len / 64),
            },
            .{
                .path = "./src/shader.glsl",
                .define = "EYEFACE_OCCLUSION",
                .group_x = app_state.voxels.side * app_state.voxels.side * app_state.voxels.side / 64,
            },
            .{
                .path = "./src/shader.glsl",
                .define = "EYEFACE_ITERATE",
                .group_x = @intCast(vertices.items.len / 64),
            },
        };

        for (pipelines, 0..) |p, i| {
            const compute_spv = blk1: {
                compiler.stage = .compute;
                const frag: utils.Glslc.Compiler.Code = .{ .path = .{
                    .main = p.path,
                    .include = &[_][]const u8{},
                    .definitions = &[_][]const u8{ "EYEFACE_COMPUTE", p.define },
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
                        break :blk1 ok;
                    },
                }
            };
            defer allocator.free(compute_spv);

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

            _ = try device.createComputePipelines(.null_handle, 1, @ptrCast(&cpci), null, @ptrCast(&pipelines[i].pipeline));
        }

        break :blk pipelines;
    };
    errdefer {
        for (compute_pipelines) |p| {
            device.destroyPipeline(p.pipeline, null);
        }
    }

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
            // .vertex_binding_description_count = 0,
            // .p_vertex_binding_descriptions = @ptrCast(&Vertex.binding_description),
            // .vertex_attribute_description_count = Vertex.attribute_description.len,
            // .p_vertex_attribute_descriptions = &Vertex.attribute_description,
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

        const depth_stencil_info = vk.PipelineDepthStencilStateCreateInfo{
            .depth_test_enable = vk.TRUE,
            .depth_write_enable = vk.TRUE,
            .depth_compare_op = .less,
            .depth_bounds_test_enable = vk.FALSE,
            .stencil_test_enable = vk.FALSE,
            .front = .{
                .fail_op = .keep,
                .pass_op = .replace,
                .depth_fail_op = .keep,
                .compare_op = .always,
                .compare_mask = 0xFF,
                .write_mask = 0xFF,
                .reference = 1,
            },
            .back = .{
                .fail_op = .keep,
                .pass_op = .replace,
                .depth_fail_op = .keep,
                .compare_op = .always,
                .compare_mask = 0xFF,
                .write_mask = 0xFF,
                .reference = 1,
            },
            .min_depth_bounds = 0.0,
            .max_depth_bounds = 1.0,
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
            .p_depth_stencil_state = &depth_stencil_info,
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
            const attachments = [_]vk.ImageView{ swapchain.swap_images[i_2].view, depth_image.view };
            fb.* = try device.createFramebuffer(&.{
                .render_pass = pass,
                .attachment_count = attachments.len,
                .p_attachments = &attachments,
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

    const uniforms = app_state.uniforms(engine.window);

    const command_buffers = blk: {
        const cmdbufs = try allocator.alloc(vk.CommandBuffer, framebuffers.len);
        errdefer allocator.free(cmdbufs);

        try device.allocateCommandBuffers(&.{
            .command_pool = pool,
            .level = .primary,
            .command_buffer_count = @intCast(cmdbufs.len),
        }, cmdbufs.ptr);
        errdefer device.freeCommandBuffers(pool, @intCast(cmdbufs.len), cmdbufs.ptr);

        const clear = [_]vk.ClearValue{
            .{
                .color = .{
                    .float_32 = app_state.background_color.gamma_correct_inv().to_buf(),
                },
            },
            .{
                .depth_stencil = .{
                    .depth = 1.0,
                    .stencil = 0,
                },
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

            for (compute_pipelines) |p| {
                device.cmdBindPipeline(cmdbuf, .compute, p.pipeline);
                device.cmdBindDescriptorSets(cmdbuf, .compute, compute_pipeline_layout, 0, 1, @ptrCast(&compute_desc_set), 0, null);
                device.cmdDispatch(cmdbuf, p.group_x, p.group_y, p.group_z);

                device.cmdPipelineBarrier(cmdbuf, .{
                    .compute_shader_bit = true,
                }, .{
                    .compute_shader_bit = true,
                }, .{}, 1, &[_]vk.MemoryBarrier{.{
                    .src_access_mask = .{
                        .shader_read_bit = true,
                        .shader_write_bit = true,
                    },
                    .dst_access_mask = .{
                        .shader_read_bit = true,
                        .shader_write_bit = true,
                    },
                }}, 0, null, 0, null);
            }

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
                .clear_value_count = clear.len,
                .p_clear_values = &clear,
            }, .@"inline");

            device.cmdBindPipeline(cmdbuf, .graphics, pipeline);
            device.cmdBindDescriptorSets(cmdbuf, .graphics, pipeline_layout, 0, 1, @ptrCast(&frag_desc_set), 0, null);
            device.cmdDraw(cmdbuf, 6, 1, 0, 0);

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
        .vertex_buffer_memory = vertex_buffer.memory,
        .vertex_buffer = vertex_buffer.buffer,
        .depth_image = depth_image.image,
        .depth_buffer_memory = depth_image.memory,
        .depth_image_view = depth_image.view,
        .voxel_buffer = voxels.voxel_buffer,
        .voxel_buffer_memory = voxels.voxel_buffer_memory,
        .occlusion_buffer = voxels.occlusion_buffer,
        .occlusion_buffer_memory = voxels.occlusion_buffer_memory,
        .screen_buffer = screen.buffer,
        .screen_buffer_memory = screen.memory,
        .screen_depth_buffer = screen_depth.buffer,
        .screen_depth_buffer_memory = screen_depth.memory,
        .descriptor_pool = desc_pool,
        .frag_descriptor_set_layout = frag_desc_set_layout,
        .frag_descriptor_set = frag_desc_set,
        .compute_descriptor_set_layout = compute_desc_set_layout,
        .compute_descriptor_set = compute_desc_set,
        .pass = pass,
        .compute_pipeline_layout = compute_pipeline_layout,
        .compute_pipelines = blk: {
            const pipelines = try allocator.alloc(vk.Pipeline, compute_pipelines.len);
            for (compute_pipelines, 0..) |p, i| {
                pipelines[i] = p.pipeline;
            }
            break :blk pipelines;
        },
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,
        .framebuffers = framebuffers,
        .command_pool = pool,
        .command_buffers = command_buffers,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
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
    defer {
        device.destroyImageView(self.depth_image_view, null);
        device.freeMemory(self.depth_buffer_memory, null);
        device.destroyImage(self.depth_image, null);
    }
    defer {
        device.destroyBuffer(self.voxel_buffer, null);
        device.freeMemory(self.voxel_buffer_memory, null);
        device.destroyBuffer(self.occlusion_buffer, null);
        device.freeMemory(self.occlusion_buffer_memory, null);
    }
    defer {
        device.destroyBuffer(self.screen_buffer, null);
        device.freeMemory(self.screen_buffer_memory, null);
    }
    defer {
        device.destroyBuffer(self.screen_depth_buffer, null);
        device.freeMemory(self.screen_depth_buffer_memory, null);
    }
    defer device.destroyDescriptorSetLayout(self.frag_descriptor_set_layout, null);
    defer device.destroyDescriptorSetLayout(self.compute_descriptor_set_layout, null);
    defer device.destroyDescriptorPool(self.descriptor_pool, null);

    defer device.destroyRenderPass(self.pass, null);
    defer device.destroyPipelineLayout(self.compute_pipeline_layout, null);
    defer {
        for (self.compute_pipelines) |p| {
            device.destroyPipeline(p, null);
        }
        allocator.free(self.compute_pipelines);
    }
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

pub fn present(
    self: *@This(),
    gui_renderer: *GuiEngine.GuiRenderer,
    ctx: *Engine.VulkanContext,
) !Swapchain.PresentState {
    const cmdbuf = self.command_buffers[self.swapchain.image_index];
    const gui_cmdbuf = gui_renderer.cmd_bufs[self.swapchain.image_index];

    return self.swapchain.present(&[_]vk.CommandBuffer{ cmdbuf, gui_cmdbuf }, ctx, &self.uniforms, &self.uniform_memory) catch |err| switch (err) {
        error.OutOfDateKHR => return .suboptimal,
        else => |narrow| return narrow,
    };
}

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
