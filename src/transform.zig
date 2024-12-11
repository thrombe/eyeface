const std = @import("std");

const utils = @import("utils.zig");

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Mat4x4 = math.Mat4x4;

pub fn TransformSet(n: u32) type {
    return extern struct {
        transforms: [n]Mat4x4 = std.mem.zeroes([n]Mat4x4),

        pub const Builder = struct {
            transforms: [n]Transforms,

            pub fn init() @This() {
                var this = std.mem.zeroes(@This());

                inline for (0..n) |i| {
                    this.transforms[i] = Transforms.init();
                }

                return this;
            }

            pub fn random(rng: std.Random) @This() {
                var this = std.mem.zeroes(@This());

                inline for (0..n) |i| {
                    this.transforms[i] = Transforms.random(rng);
                }

                return this;
            }

            pub fn mix(self: *const @This(), other: *const @This(), t: f32) @This() {
                var this = std.mem.zeroes(@This());

                inline for (0..n) |i| {
                    this.transforms[i] = self.transforms[i].mix(&other.transforms[i], t);
                }

                return this;
            }

            pub fn build(self: *const @This()) TransformSet(n) {
                var this = std.mem.zeroes(TransformSet(n));

                inline for (0..n) |i| {
                    this.transforms[i] = self.transforms[i].combine();
                }

                return this;
            }

            pub const Transforms = struct {
                translate: Vec4,
                shear: struct {
                    x: Vec4,
                    y: Vec4,
                    z: Vec4,
                },
                scale: Vec4,
                rot: Vec4,

                pub fn init() @This() {
                    return .{
                        .translate = Vec4{},
                        .shear = .{
                            .x = Vec4{},
                            .y = Vec4{},
                            .z = Vec4{},
                        },
                        .scale = Vec4.splat3(1),
                        .rot = Vec4.quat_identity_rot(),
                    };
                }

                pub fn combine(self: *const @This()) Mat4x4 {
                    var mat = Mat4x4.identity();
                    mat = mat.mul_mat(Mat4x4.scaling_mat(self.scale));
                    mat = mat.mul_mat(Mat4x4.rot_mat_euler_angles(self.rot));
                    const shearx = Mat4x4.shear_mat(.{
                        .y = self.shear.x.y,
                        .z = self.shear.x.z,
                    }, .{}, .{});
                    const sheary = Mat4x4.shear_mat(.{}, .{
                        .x = self.shear.y.x,
                        .z = self.shear.y.z,
                    }, .{});
                    const shearz = Mat4x4.shear_mat(.{}, .{}, .{
                        .x = self.shear.z.x,
                        .y = self.shear.z.y,
                    });
                    mat = mat.mul_mat(shearz);
                    mat = mat.mul_mat(sheary);
                    mat = mat.mul_mat(shearx);
                    mat = mat.mul_mat(Mat4x4.translation_mat(self.translate));
                    return mat;
                }

                pub fn mix(self: *const @This(), other: *const @This(), t: f32) @This() {
                    return .{
                        .translate = self.translate.mix(other.translate, t),
                        .shear = .{
                            .x = self.shear.x.mix(other.shear.x, t),
                            .y = self.shear.y.mix(other.shear.y, t),
                            .z = self.shear.z.mix(other.shear.z, t),
                        },
                        .scale = self.scale.mix(other.scale, t),
                        .rot = self.rot.mix(other.rot, t),
                    };
                }
            };

            pub const Generator = struct {
                scale: Vec3Constraints = Vec3Constraints.splat(.{ .min = 0.4, .max = 0.75 }),
                rot: Vec3Constraints = Vec3Constraints.splat(.{
                    .min = -std.math.pi / 6.0,
                    .max = std.math.pi / 4.0,
                }),
                translate: Vec3Constraints = Vec3Constraints.splat(.{ .min = -1.0, .max = 1.0 }),
                shear: ShearConstraints = .{},

                pub const Constraints = math.Rng.Constraints;
                pub const ShearConstraints = struct {
                    x: struct {
                        y: Constraints = .{ .min = -0.2, .max = 0.2 },
                        z: Constraints = .{ .min = -0.2, .max = 0.2 },
                    } = .{},
                    y: struct {
                        x: Constraints = .{ .min = -0.2, .max = 0.2 },
                        z: Constraints = .{ .min = -0.2, .max = 0.2 },
                    } = .{},
                    z: struct {
                        x: Constraints = .{ .min = -0.2, .max = 0.2 },
                        y: Constraints = .{ .min = -0.2, .max = 0.2 },
                    } = .{},
                };
                pub const Vec3Constraints = struct {
                    x: Constraints = .{},
                    y: Constraints = .{},
                    z: Constraints = .{},

                    pub fn splat(constraint: Constraints) @This() {
                        return .{
                            .x = constraint,
                            .y = constraint,
                            .z = constraint,
                        };
                    }

                    pub fn random(self: *const @This(), _rng: std.Random) Vec4 {
                        const rng = math.Rng.init(_rng);
                        return .{
                            .x = rng.with2(self.x).next(),
                            .y = rng.with2(self.x).next(),
                            .z = rng.with2(self.x).next(),
                        };
                    }
                };

                pub fn generate(self: *const @This(), _rng: std.Random) Builder {
                    var builder = std.mem.zeroes(Builder);
                    const rng = math.Rng.init(_rng);

                    inline for (0..n) |i| {
                        builder.transforms[i] = .{
                            .translate = self.translate.random(_rng),
                            .scale = self.scale.random(_rng),
                            .rot = self.rot.random(_rng),
                            .shear = .{
                                .x = .{
                                    .y = rng.with2(self.shear.x.y).next(),
                                    .z = rng.with2(self.shear.x.z).next(),
                                },
                                .y = .{
                                    .x = rng.with2(self.shear.y.x).next(),
                                    .z = rng.with2(self.shear.y.z).next(),
                                },
                                .z = .{
                                    .x = rng.with2(self.shear.z.x).next(),
                                    .y = rng.with2(self.shear.z.y).next(),
                                },
                            },
                        };
                    }

                    return builder;
                }
            };
        };
    };
}

pub fn sirpinski_pyramid() TransformSet(5).Builder {
    var this = TransformSet(5).Builder.init();

    const s = Vec4.splat3(0.5);

    inline for (0..5) |i| {
        this.transforms[i].scale = s;
    }

    this.transforms[0].translate = .{ .x = 0.0, .y = 1.0, .z = 0.0 };
    this.transforms[1].translate = .{ .x = 1.0, .y = -1.0, .z = 1.0 };
    this.transforms[2].translate = .{ .x = 1.0, .y = -1.0, .z = -1.0 };
    this.transforms[3].translate = .{ .x = -1.0, .y = -1.0, .z = 1.0 };
    this.transforms[4].translate = .{ .x = -1.0, .y = -1.0, .z = -1.0 };

    return this;
}
