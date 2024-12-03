const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // generate vulkan bindings from vk.xml and create a zig module from the generated code
    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");
    const vk_gen = b.dependency("vulkan_zig", .{}).artifact("vulkan-zig-generator");
    const vk_generate_cmd = b.addRunArtifact(vk_gen);
    vk_generate_cmd.addFileArg(registry);
    const vulkan_zig = b.addModule("vulkan-zig", .{
        .root_source_file = vk_generate_cmd.addOutputFileArg("vk.zig"),
    });

    // const glfw = b.dependency("glfw", .{
    //     .target = target,
    //     .optimize = optimize,
    // });

    const cimgui_dep = b.dependency("cimgui", .{
        .target = target,
        .optimize = optimize,
        .platform = .GLFW,
        .renderer = .Vulkan,
    });

    const exe = b.addExecutable(.{
        .name = "eyeface",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("vulkan", vulkan_zig);
    exe.linkLibrary(cimgui_dep.artifact("cimgui"));
    exe.addIncludePath(cimgui_dep.path("dcimgui/backends"));
    exe.linkSystemLibrary("fswatch");
    exe.linkSystemLibrary2("ImageMagick", .{});
    exe.linkSystemLibrary2("MagickWand", .{});
    exe.linkSystemLibrary2("MagickCore", .{});

    // exe.root_module.linkLibrary(glfw.artifact("glfw"));
    // @import("glfw").addPaths(&exe.root_module);
    // exe.linkSystemLibrary("glfw");
    exe.linkLibC();
    b.installArtifact(exe);

    const vert_cmd = b.addSystemCommand(&.{
        "glslc",
        "--target-env=vulkan1.2",
        "-o",
    });
    const vert_spv = vert_cmd.addOutputFileArg("vert.spv");
    vert_cmd.addFileArg(b.path("tmp/vulkan-zig/examples/shaders/triangle.vert"));
    exe.root_module.addAnonymousImport("vertex_shader", .{
        .root_source_file = vert_spv,
    });

    const frag_cmd = b.addSystemCommand(&.{
        "glslc",
        "--target-env=vulkan1.2",
        "-o",
    });
    const frag_spv = frag_cmd.addOutputFileArg("frag.spv");
    frag_cmd.addFileArg(b.path("tmp/vulkan-zig/examples/shaders/triangle.frag"));
    exe.root_module.addAnonymousImport("fragment_shader", .{
        .root_source_file = frag_spv,
    });

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
