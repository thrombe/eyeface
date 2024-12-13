const std = @import("std");

const transform = @import("transform.zig");

const Engine = @import("engine.zig");
const c = Engine.c;

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Mat4x4 = math.Mat4x4;

const Renderer = @import("renderer.zig");

pub const AppState = struct {
    pos: Vec4,
    mouse: extern struct { x: i32 = 0, y: i32 = 0, left: bool = false, right: bool = false } = .{},
    pitch: f32 = 0,
    yaw: f32 = 0,

    speed: f32 = 1.0,
    sensitivity: f32 = 1.0,

    frame: u32 = 0,
    time: f32 = 0,
    deltatime: f32 = 0,

    transform_generator: Uniforms.TransformSet.Builder.Generator = .{},
    transforms: Uniforms.TransformSet.Builder,
    target_transforms: Uniforms.TransformSet.Builder,

    t: f32 = 0,
    lambda: f32 = 1.0,
    pause_t: bool = false,
    pause_generator: bool = false,
    points_x_64: u32 = 50000,
    iterations: u32 = 20,
    voxelization_points_x_64: u32 = 50000,
    voxelization_iterations: u32 = 4,
    reduction_points_x_64: u32 = 50000,

    background_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#282828ff"),
    occlusion_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#401a1aff"),
    sparse_color: Vec4 = math.ColorParse.hex_xyzw(Vec4, "#e0e7cdff"),
    occlusion_multiplier: f32 = 1.25,
    occlusion_attenuation: f32 = 2.0,

    voxels: struct {
        // world space coords of center of the the voxel grid
        center: Vec4 = .{},
        // number of voxels along 1 edge (side ** 3 is the entire volume)
        side: u32 = 300,
    } = .{},

    rng: std.Random.Xoshiro256,

    pub const Uniforms = Renderer.Uniforms;

    pub const constants = struct {
        const pitch_min = -std.math.pi / 2.0 + 0.1;
        const pitch_max = std.math.pi / 2.0 - 0.1;
        const up = Vec4{ .y = -1 };
        const fwd = Vec4{ .z = 1 };
        const right = Vec4{ .x = 1 };
    };

    pub fn init(window: *Engine.Window) @This() {
        const pos = Vec4{ .z = -5 };
        const mouse = window.poll_mouse();

        var rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));

        const generator = Uniforms.TransformSet.Builder.Generator{};

        return .{
            .pos = pos,
            .mouse = .{ .x = mouse.x, .y = mouse.y, .left = mouse.left },
            .transforms = transform.sirpinski_pyramid(),
            .target_transforms = generator.generate(rng.random()),
            .transform_generator = generator,
            .rng = rng,
        };
    }

    pub fn tick(self: *@This(), lap: u64, window: *Engine.Window) void {
        const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        // std.debug.print("fps: {d}\n", .{@as(u32, @intFromFloat(1.0 / delta))});
        const w = window.is_pressed(c.GLFW_KEY_W);
        const a = window.is_pressed(c.GLFW_KEY_A);
        const s = window.is_pressed(c.GLFW_KEY_S);
        const d = window.is_pressed(c.GLFW_KEY_D);
        const mouse = window.poll_mouse();

        if (mouse.left) {
            self.yaw += @as(f32, @floatFromInt(mouse.x - self.mouse.x)) * self.sensitivity * delta;
            self.pitch -= @as(f32, @floatFromInt(mouse.y - self.mouse.y)) * self.sensitivity * delta;
            self.pitch = std.math.clamp(self.pitch, constants.pitch_min, constants.pitch_max);
        }

        self.mouse.left = mouse.left;
        self.mouse.x = mouse.x;
        self.mouse.y = mouse.y;

        const rot = self.rot_quat();
        const fwd = rot.rotate_vector(constants.fwd);
        const right = rot.rotate_vector(constants.right);

        if (w) {
            self.pos = self.pos.add(fwd.scale(delta * self.speed));
        }
        if (a) {
            self.pos = self.pos.sub(right.scale(delta * self.speed));
        }
        if (s) {
            self.pos = self.pos.sub(fwd.scale(delta * self.speed));
        }
        if (d) {
            self.pos = self.pos.add(right.scale(delta * self.speed));
        }

        self.frame += 1;
        self.time += delta;
        self.deltatime = delta;

        if (!self.pause_t) {
            // - [Lerp smoothing is broken](https://youtu.be/LSNQuFEDOyQ?si=-_bGNwqZFC_j5dJF&t=3012)
            const e = 1.0 - std.math.exp(-self.lambda * delta);
            self.transforms = self.transforms.mix(&self.target_transforms, e);
            self.t = std.math.lerp(self.t, 1.0, e);
        }
        if (1.0 - self.t < 0.01 and !self.pause_generator) {
            self.t = 0;
            self.target_transforms = self.transform_generator.generate(self.rng.random());
        }
    }

    fn rot_quat(self: *const @This()) Vec4 {
        var rot = Vec4.quat_identity_rot();
        rot = rot.quat_mul(Vec4.quat_angle_axis(self.pitch, constants.right));
        rot = rot.quat_mul(Vec4.quat_angle_axis(self.yaw, constants.up));
        rot = rot.quat_conjugate();
        return rot;
    }

    pub fn uniforms(self: *const @This(), window: *Engine.Window) Uniforms {
        const rot = self.rot_quat();
        const up = rot.rotate_vector(constants.up);
        const fwd = rot.rotate_vector(constants.fwd);

        const projection_matrix = Mat4x4.perspective_projection(window.extent.height, window.extent.width, 0.01, 100.0, std.math.pi / 3.0);
        const view_matrix = Mat4x4.view(self.pos, fwd, up);
        const world_to_screen = projection_matrix.mul_mat(view_matrix);

        const transforms = self.transforms.build();
        return .{
            .transforms = transforms,
            .world_to_screen = world_to_screen,
            .eye = self.pos,
            .mouse = .{
                .x = self.mouse.x,
                .y = self.mouse.y,
                .left = @intCast(@intFromBool(self.mouse.left)),
                .right = @intCast(@intFromBool(self.mouse.right)),
            },
            .voxel_grid_center = self.voxels.center,
            .voxel_grid_side = self.voxels.side,
            .occlusion_color = self.occlusion_color,
            .sparse_color = self.sparse_color,
            .background_color = self.background_color,
            .occlusion_multiplier = self.occlusion_multiplier,
            .occlusion_attenuation = self.occlusion_attenuation,
            .points = self.points_x_64 * 64,
            .iterations = self.iterations,
            .voxelization_points = self.voxelization_points_x_64 * 64,
            .voxelization_iterations = self.voxelization_iterations,
            .reduction_points = self.reduction_points_x_64 * 64,
            .frame = self.frame,
            .time = self.time,
            .deltatime = self.deltatime,
            .lambda = self.lambda,
            .width = window.extent.width,
            .height = window.extent.height,
        };
    }
};

pub const GuiState = struct {
    const TransformSet = Renderer.Uniforms.TransformSet;
    const Constraints = TransformSet.Builder.Generator.Constraints;
    const ShearConstraints = TransformSet.Builder.Generator.ShearConstraints;
    const Vec3Constraints = TransformSet.Builder.Generator.Vec3Constraints;

    pub fn tick(self: *@This(), state: *AppState, lap: u64) void {
        const delta = @as(f32, @floatFromInt(lap)) / @as(f32, @floatFromInt(std.time.ns_per_s));
        const generator = &state.transform_generator;

        c.ImGui_SetNextWindowPos(.{ .x = 5, .y = 5 }, c.ImGuiCond_Once);
        defer c.ImGui_End();
        if (c.ImGui_Begin("SIKE", null, c.ImGuiWindowFlags_None)) {
            c.ImGui_Text("Application average %.3f ms/frame (%.1f FPS)", delta, 1.0 / delta);

            c.ImGui_Text("State");
            self.editState(state);

            c.ImGui_Text("Scale");
            self.editVec3Constraints("Scale", &generator.scale);

            c.ImGui_Text("Rotation");
            self.editVec3Constraints("Rotation", &generator.rot);

            c.ImGui_Text("Translation");
            self.editVec3Constraints("Translation", &generator.translate);

            c.ImGui_Text("Shear");
            self.editShear("Shear", &generator.shear);
        }
    }

    fn editState(self: *@This(), state: *AppState) void {
        _ = self;

        _ = c.ImGui_SliderFloat("Speed", &state.speed, 0.1, 10.0);
        _ = c.ImGui_SliderFloat("Sensitivity", &state.sensitivity, 0.1, 10.0);

        _ = c.ImGui_SliderInt("points (x 64)", @ptrCast(&state.points_x_64), 0, 1000000);
        state.voxelization_points_x_64 = @min(state.voxelization_points_x_64, state.points_x_64);
        _ = c.ImGui_SliderInt("voxelization points (x 64)", @ptrCast(&state.voxelization_points_x_64), 0, @intCast(state.points_x_64));
        state.reduction_points_x_64 = @min(state.reduction_points_x_64, state.points_x_64);
        _ = c.ImGui_SliderInt("reduction points (x 64)", @ptrCast(&state.reduction_points_x_64), 0, @intCast(state.points_x_64));
        _ = c.ImGui_SliderInt("iterations", @ptrCast(&state.iterations), 0, 100);
        _ = c.ImGui_SliderInt("voxelization iterations", @ptrCast(&state.voxelization_iterations), 0, 20);

        _ = c.ImGui_ColorEdit3("Background Color", @ptrCast(&state.background_color), c.ImGuiColorEditFlags_Float);
        _ = c.ImGui_ColorEdit3("Occlusion Color", @ptrCast(&state.occlusion_color), c.ImGuiColorEditFlags_Float);
        _ = c.ImGui_ColorEdit3("Sparse Color", @ptrCast(&state.sparse_color), c.ImGuiColorEditFlags_Float);

        _ = c.ImGui_SliderFloat("Occlusion Multiplier", &state.occlusion_multiplier, 0.1, 10.0);
        _ = c.ImGui_SliderFloat("Occlusion Attenuation", &state.occlusion_attenuation, 0.1, 10.0);

        _ = c.ImGui_SliderFloat("Lambda", &state.lambda, 0.1, 25.0);
        _ = c.ImGui_Checkbox("Pause t (pause_t)", &state.pause_t);
        _ = c.ImGui_Checkbox("Pause Generator (pause_generator)", &state.pause_generator);

        _ = c.ImGui_DragFloat3("Voxels Center", @ptrCast(&state.voxels.center));

        _ = c.ImGui_SliderInt("Voxel Side", @ptrCast(&state.voxels.side), 1, 500);
    }

    fn editVec3Constraints(self: *@This(), comptime label: [:0]const u8, constraints: *Vec3Constraints) void {
        c.ImGui_PushID(label);
        self.editConstraint(".X", &constraints.x);
        self.editConstraint(".Y", &constraints.y);
        self.editConstraint(".Z", &constraints.z);
        c.ImGui_PopID();
    }

    fn editConstraint(self: *@This(), comptime label: [:0]const u8, constraint: *Constraints) void {
        _ = self;

        const width = 75.0;

        c.ImGui_SetNextItemWidth(width);
        _ = c.ImGui_DragFloat(label ++ " Min", &constraint.min);
        c.ImGui_SameLine();
        c.ImGui_SetNextItemWidth(width);
        _ = c.ImGui_DragFloat(label ++ " Max", &constraint.max);
        c.ImGui_SameLine();
        _ = c.ImGui_Checkbox(label ++ " Flip Sign", &constraint.flip_sign);
    }

    fn editShear(self: *@This(), comptime label: [:0]const u8, shear: *ShearConstraints) void {
        c.ImGui_PushID(label);
        self.editConstraint(".X.y", &shear.x.y);
        self.editConstraint(".X.z", &shear.x.z);
        self.editConstraint(".Y.x", &shear.y.x);
        self.editConstraint(".Y.z", &shear.y.z);
        self.editConstraint(".Z.x", &shear.z.x);
        self.editConstraint(".Z.y", &shear.z.y);
        c.ImGui_PopID();
    }
};
