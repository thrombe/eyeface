const std = @import("std");
const main = @import("main.zig");
const allocator = main.allocator;

pub const Rng = struct {
    rng: std.Random,
    constraints: Constraints = .{},

    const Constraints = struct {
        min: f32 = -1,
        max: f32 = 1,
        flip_sign: bool = false,
    };

    pub fn init(rng: std.Random) @This() {
        return .{ .rng = rng };
    }

    pub fn next(self: *const @This()) f32 {
        var rn = self.rng.float(f32);
        rn = self.constraints.min + rn * (self.constraints.max - self.constraints.min);

        if (self.constraints.flip_sign) {
            if (self.rng.boolean()) {
                rn *= -1;
            }
        }

        return rn;
    }

    pub fn with(self: *const @This(), c: struct {
        min: ?f32 = null,
        max: ?f32 = null,
        flip_sign: ?bool = null,
    }) @This() {
        var this = self.*;
        if (c.min) |min| {
            this.constraints.min = min;
        }
        if (c.max) |max| {
            this.constraints.max = max;
        }
        if (c.flip_sign) |flip| {
            this.constraints.flip_sign = flip;
        }
        return this;
    }
};

pub const Vec4 = extern struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,

    pub fn dot(self: *const @This(), other: @This()) f32 {
        return self.x * other.x +
            self.y * other.y +
            self.z * other.z +
            self.w * other.w;
    }

    pub fn cross(self: *const @This(), other: @This()) @This() {
        return .{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
            .w = 0,
        };
    }

    pub fn mul(self: *const @This(), other: @This()) @This() {
        return .{
            .x = self.x * other.x,
            .y = self.y * other.y,
            .z = self.z * other.z,
            .w = self.w * other.w,
        };
    }

    pub fn add(self: *const @This(), other: @This()) @This() {
        return .{
            .x = self.x + other.x,
            .y = self.y + other.y,
            .z = self.z + other.z,
            .w = self.w + other.w,
        };
    }

    pub fn sub(self: *const @This(), other: @This()) @This() {
        return .{
            .x = self.x - other.x,
            .y = self.y - other.y,
            .z = self.z - other.z,
            .w = self.w - other.w,
        };
    }

    pub fn scale(self: *const @This(), s: f32) @This() {
        return .{ .x = self.x * s, .y = self.y * s, .z = self.z * s, .w = self.w * s };
    }

    pub fn mix(self: *const @This(), other: @This(), t: f32) @This() {
        return .{
            .x = std.math.lerp(self.x, other.x, t),
            .y = std.math.lerp(self.y, other.y, t),
            .z = std.math.lerp(self.z, other.z, t),
            .w = std.math.lerp(self.w, other.w, t),
        };
    }

    pub fn splat3(t: f32) @This() {
        return .{ .x = t, .y = t, .z = t };
    }

    pub fn splat4(t: f32) @This() {
        return .{ .x = t, .y = t, .z = t, .w = t };
    }

    pub fn normalize3D(self: *const @This()) @This() {
        var this = self.*;
        this.w = 0;

        const size = @sqrt(this.dot(this));
        return .{
            .x = self.x / size,
            .y = self.y / size,
            .z = self.z / size,
            .w = self.w,
        };
    }

    pub fn normalize4D(self: *const @This()) @This() {
        const size = @sqrt(self.dot(self.*));
        return .{
            .x = self.x / size,
            .y = self.y / size,
            .z = self.z / size,
            .w = self.w / size,
        };
    }

    pub fn quat_identity_rot() @This() {
        return .{ .w = 1 };
    }

    pub fn quat_euler_angles(pitch: f32, yaw: f32) @This() {
        // No roll is used, only pitch and yaw
        const half_pitch = pitch * 0.5;
        const half_yaw = yaw * 0.5;

        const cos_pitch = @cos(half_pitch);
        const sin_pitch = @sin(half_pitch);
        const cos_yaw = @cos(half_yaw);
        const sin_yaw = @sin(half_yaw);

        return .{
            .w = cos_pitch * cos_yaw,
            .x = sin_pitch * cos_yaw,
            .y = cos_pitch * sin_yaw,
            .z = -sin_pitch * sin_yaw, // Negative for correct rotation direction
        };
    }

    pub fn quat_angle_axis(angle: f32, axis: Vec4) @This() {
        // - [Visualizing quaternions, an explorable video series](https://eater.net/quaternions)
        const s = @sin(angle / 2.0);
        var q = axis.normalize3D().scale(s);
        q.w = @cos(angle / 2.0);
        return q;
    }

    // mult from the right means applying that rotation first.
    pub fn quat_mul(self: *const @This(), other: @This()) @This() {
        return .{
            .w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            .x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            .y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z,
            .z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x,
        };
    }

    // - [How to Use Quaternions - YouTube](https://www.youtube.com/watch?v=bKd2lPjl92c)
    // use this for rotation relative to world axes
    pub fn quat_global_rot(self: *const @This(), other: @This()) @This() {
        return other.quat_mul(self.*);
    }

    // use this for rotations relative to player's fwd, right, up as the axes
    pub fn quat_local_rot(self: *const @This(), other: @This()) @This() {
        return self.quat_mul(other);
    }

    pub fn quat_conjugate(self: *const @This()) @This() {
        return .{ .w = self.w, .x = -self.x, .y = -self.y, .z = -self.z };
    }

    pub fn rotate_vector(self: *const @This(), v: Vec4) Vec4 {
        const qv = .{ .w = 0, .x = v.x, .y = v.y, .z = v.z };
        const q_conjugate = self.quat_conjugate();
        const q_result = self.quat_mul(qv).quat_mul(q_conjugate);
        return Vec4{ .x = q_result.x, .y = q_result.y, .z = q_result.z };
    }

    pub fn to_buf(self: *const @This()) [4]f32 {
        return .{ self.x, self.y, self.z, self.w };
    }

    pub fn gamma_correct_inv(self: *const @This()) @This() {
        // const p: f32 = 1.0 / 2.2;
        const p: f32 = 2.2;
        return .{
            .x = std.math.pow(f32, self.x, p),
            .y = std.math.pow(f32, self.y, p),
            .z = std.math.pow(f32, self.z, p),
            .w = std.math.pow(f32, self.w, p),
        };
    }

    pub const random = struct {
        pub fn vec3(rng: *const Rng) Vec4 {
            return .{
                .x = rng.next(),
                .y = rng.next(),
                .z = rng.next(),
            };
        }

        pub fn vec4(rng: *const Rng) Vec4 {
            return .{
                .x = rng.next(),
                .y = rng.next(),
                .z = rng.next(),
                .w = rng.next(),
            };
        }
    };
};

// - [Matrix storage](https://github.com/hexops/machengine.org/blob/0aab00137dc3d1098e5237e2bee124e0ef9fbc17/content/docs/math/matrix-storage.md)
// vulkan wants | V1, V2, V3, V4 | (columns contiguous in memory).
// so we need to store matrix in transposed form
//
// all computation below is performed assuming right associative multiplication
// and uses column vectors (even though it is stored as row vectors in the struct)
// self.data[0] is 1 vector
//
pub const Mat4x4 = extern struct {
    data: [4]Vec4 = std.mem.zeroes([4]Vec4),

    pub fn mul_vec4(self: *const @This(), v: Vec4) Vec4 {
        const this = self.transpose();
        return .{
            .x = this.data[0].dot(v),
            .y = this.data[1].dot(v),
            .z = this.data[2].dot(v),
            .w = this.data[3].dot(v),
        };
    }

    pub fn mul_mat(self: *const @This(), o: @This()) @This() {
        const this = self.transpose();
        return .{ .data = .{
            .{
                .x = this.data[0].dot(o.data[0]),
                .y = this.data[1].dot(o.data[0]),
                .z = this.data[2].dot(o.data[0]),
                .w = this.data[3].dot(o.data[0]),
            },
            .{
                .x = this.data[0].dot(o.data[1]),
                .y = this.data[1].dot(o.data[1]),
                .z = this.data[2].dot(o.data[1]),
                .w = this.data[3].dot(o.data[1]),
            },
            .{
                .x = this.data[0].dot(o.data[2]),
                .y = this.data[1].dot(o.data[2]),
                .z = this.data[2].dot(o.data[2]),
                .w = this.data[3].dot(o.data[2]),
            },
            .{
                .x = this.data[0].dot(o.data[3]),
                .y = this.data[1].dot(o.data[3]),
                .z = this.data[2].dot(o.data[3]),
                .w = this.data[3].dot(o.data[3]),
            },
        } };
    }

    pub fn transpose(self: *const @This()) @This() {
        return .{ .data = .{
            .{ .x = self.data[0].x, .y = self.data[1].x, .z = self.data[2].x, .w = self.data[3].x },
            .{ .x = self.data[0].y, .y = self.data[1].y, .z = self.data[2].y, .w = self.data[3].y },
            .{ .x = self.data[0].z, .y = self.data[1].z, .z = self.data[2].z, .w = self.data[3].z },
            .{ .x = self.data[0].w, .y = self.data[1].w, .z = self.data[2].w, .w = self.data[3].w },
        } };
    }

    pub fn mix(self: *const @This(), other: *const @This(), t: f32) @This() {
        var this = std.mem.zeroes(@This());
        inline for (0..self.data.len) |i| {
            this.data[i] = self.data[i].mix(other.data[i], t);
        }
        return this;
    }

    pub fn identity() @This() {
        return .{ .data = .{
            .{ .x = 1, .y = 0, .z = 0, .w = 0 },
            .{ .x = 0, .y = 1, .z = 0, .w = 0 },
            .{ .x = 0, .y = 0, .z = 1, .w = 0 },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        } };
    }

    pub fn perspective_projection(height: u32, width: u32, near: f32, far: f32, fov: f32) @This() {
        // - [Perspective Projection](https://www.youtube.com/watch?v=U0_ONQQ5ZNM)
        var self = @This(){};

        const a = @as(f32, @floatFromInt(height)) / @as(f32, @floatFromInt(width));
        const f = 1.0 / @tan(fov / 2);
        const l = far / (far - near);

        self.data[0].x = a * f;
        self.data[1].y = f;
        self.data[2].z = l;
        self.data[2].w = 1;
        self.data[3].z = -l * near;

        return self;
    }

    pub fn view(eye: Vec4, at: Vec4, _up: Vec4) @This() {
        // - [» Deriving the View Matrix](https://twodee.org/blog/17560)

        const front = at.normalize3D();
        const up = _up.normalize3D();
        const right = front.cross(up);

        const translate_inv = Mat4x4{ .data = .{
            .{ .x = 1, .y = 0, .z = 0, .w = -eye.x },
            .{ .x = 0, .y = 1, .z = 0, .w = -eye.y },
            .{ .x = 0, .y = 0, .z = 1, .w = -eye.z },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        } };

        return (Mat4x4{ .data = .{
            right,
            up,
            front,
            .{ .w = 1 },
        } }).transpose().mul_mat(translate_inv.transpose());
    }

    pub fn scaling_mat(vec3: Vec4) @This() {
        return .{ .data = .{
            .{ .x = vec3.x },
            .{ .y = vec3.y },
            .{ .z = vec3.z },
            .{ .w = 1 },
        } };
    }

    pub fn translation_mat(vec3: Vec4) @This() {
        return .{ .data = .{
            .{ .x = 1 },
            .{ .y = 1 },
            .{ .z = 1 },
            .{ .x = vec3.x, .y = vec3.y, .z = vec3.z, .w = 1 },
        } };
    }

    pub fn rot_mat_from_quat(rot: Vec4) @This() {
        const x = Vec4{ .x = 1 };
        const y = Vec4{ .y = 1 };
        const z = Vec4{ .z = 1 };

        return .{ .data = .{
            rot.rotate_vector(x),
            rot.rotate_vector(y),
            rot.rotate_vector(z),
            .{ .w = 1 },
        } };
    }

    // - [3D Shearing Transformation](https://www.geeksforgeeks.org/computer-graphics-3d-shearing-transformation/)
    pub fn shear_mat(
        x: struct { y: f32 = 0, z: f32 = 0 },
        y: struct { x: f32 = 0, z: f32 = 0 },
        z: struct { x: f32 = 0, y: f32 = 0 },
    ) @This() {
        return (@This(){ .data = .{
            .{ .x = 1, .y = x.y, .z = x.z },
            .{ .x = y.x, .y = 1, .z = y.z },
            .{ .x = z.x, .y = z.y, .z = 1 },
            .{ .w = 1 },
        } }).transpose();
    }

    pub const random = struct {
        // there's no point in having constrained random numbers for this
        pub fn rot(rng: *const Rng) Mat4x4 {
            var q = Vec4{
                .x = (rng.rng.float(f32) - 0.5),
                .y = (rng.rng.float(f32) - 0.5),
                .z = (rng.rng.float(f32) - 0.5),
                .w = (rng.rng.float(f32) - 0.5),
            };
            q = q.normalize4D();

            const r = Mat4x4.rot_mat_from_quat(q);
            return r;
        }

        pub fn translate(rng: *const Rng) Mat4x4 {
            return .{ .data = .{
                .{ .x = 1 }, .{ .y = 1 }, .{ .z = 1 }, .{
                    .x = rng.next(),
                    .y = rng.next(),
                    .z = rng.next(),
                    .w = 1,
                },
            } };
        }

        pub fn scale(rng: *const Rng) Mat4x4 {
            return .{
                .data = .{
                    .{ .x = rng.next() },
                    .{ .y = rng.next() },
                    .{ .z = rng.next() },
                    .{ .w = 1 },
                },
            };
        }

        pub fn shear(rng: *const Rng) Mat4x4 {
            return .{ .data = .{
                .{ .x = 1, .y = rng.next(), .z = rng.next() },
                .{ .x = rng.next(), .y = 1, .z = rng.next() },
                .{ .x = rng.next(), .y = rng.next(), .z = 1 },
                .{ .w = 1 },
            } };
        }
    };
};

pub const ColorParse = struct {
    pub fn hex_rgba(typ: type, comptime hex: []const u8) typ {
        if (hex.len != 9 or hex[0] != '#') {
            @compileError("invalid color");
        }

        return .{
            .r = @as(f32, @floatFromInt(parseHex(hex[1], hex[2]))) / 255.0,
            .g = @as(f32, @floatFromInt(parseHex(hex[3], hex[4]))) / 255.0,
            .b = @as(f32, @floatFromInt(parseHex(hex[5], hex[6]))) / 255.0,
            .a = @as(f32, @floatFromInt(parseHex(hex[7], hex[8]))) / 255.0,
        };
    }

    pub fn hex_xyzw(typ: type, comptime hex: []const u8) typ {
        if (hex.len != 9 or hex[0] != '#') {
            @compileError("invalid color");
        }

        return .{
            .x = @as(f32, @floatFromInt(parseHex(hex[1], hex[2]))) / 255.0,
            .y = @as(f32, @floatFromInt(parseHex(hex[3], hex[4]))) / 255.0,
            .z = @as(f32, @floatFromInt(parseHex(hex[5], hex[6]))) / 255.0,
            .w = @as(f32, @floatFromInt(parseHex(hex[7], hex[8]))) / 255.0,
        };
    }

    fn parseHex(comptime high: u8, comptime low: u8) u8 {
        return (hexDigitToInt(high) << 4) | hexDigitToInt(low);
    }

    fn hexDigitToInt(comptime digit: u8) u8 {
        if (digit >= '0' and digit <= '9') {
            return digit - '0';
        } else if (digit >= 'a' and digit <= 'f') {
            return digit - 'a' + 10;
        } else if (digit >= 'A' and digit <= 'F') {
            return digit - 'A' + 10;
        }
        @compileError("invalid hex digit");
    }
};

// assumes ok has ok.deinit()
pub fn Result(ok: type, err_typ: type) type {
    return union(enum) {
        Ok: ok,
        Err: Error,

        pub const Error = struct {
            err: err_typ,
            msg: []u8,

            // pub fn owned(err: err_typ, msg: []const u8) !@This() {
            //     return .{
            //         .err = err,
            //         .msg = try allocator.dupe(u8, msg),
            //     };
            // }
            // pub fn deinit(self: *@This()) void {
            //     allocator.free(self.msg);
            // }
        };

        // pub fn deinit(self: *@This()) void {
        //     switch (self) {
        //         .Ok => |res| {
        //             if (std.meta.hasMethod(ok, "deinit")) {
        //                 res.deinit();
        //             }
        //         },
        //         .Err => |err| {
        //             err.deinit();
        //         },
        //     }
        // }
    };
}

pub fn Deque(typ: type) type {
    return struct {
        allocator: std.mem.Allocator,
        buffer: []typ,
        size: usize,

        // fill this index next
        front: usize, // at
        back: usize, // one to the right

        pub fn init(alloc: std.mem.Allocator) !@This() {
            const len = 32;
            const buffer = try alloc.alloc(typ, len);
            return .{
                .allocator = alloc,
                .buffer = buffer,
                .front = 0,
                .back = 0,
                .size = 0,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.buffer);
        }

        pub fn push_front(self: *@This(), value: typ) !void {
            if (self.size == self.buffer.len) {
                try self.resize();
                return self.push_front(value) catch unreachable;
            }
            self.front = (self.front + self.buffer.len - 1) % self.buffer.len;
            self.buffer[self.front] = value;
            self.size += 1;
        }

        pub fn push_back(self: *@This(), value: typ) !void {
            if (self.size == self.buffer.len) {
                try self.resize();
                return self.push_back(value) catch unreachable;
            }
            self.buffer[self.back] = value;
            self.back = (self.back + 1) % self.buffer.len;
            self.size += 1;
        }

        pub fn pop_front(self: *@This()) ?typ {
            if (self.size == 0) {
                return null;
            }
            const value = self.buffer[self.front];
            self.front = (self.front + 1) % self.buffer.len;
            self.size -= 1;
            return value;
        }

        pub fn pop_back(self: *@This()) ?typ {
            if (self.size == 0) {
                return null;
            }
            self.back = (self.back + self.buffer.len - 1) % self.buffer.len;
            const value = self.buffer[self.back];
            self.size -= 1;
            return value;
        }

        pub fn peek_front(self: *@This()) ?*const typ {
            if (self.size == 0) {
                return null;
            }
            return &self.buffer[self.front];
        }

        pub fn peek_back(self: *@This()) ?*const typ {
            if (self.size == 0) {
                return null;
            }
            const back = (self.back + self.buffer.len - 1) % self.buffer.len;
            return &self.buffer[back];
        }

        pub fn is_empty(self: *@This()) bool {
            return self.size == 0;
        }

        fn resize(self: *@This()) !void {
            std.debug.assert(self.size == self.buffer.len);

            const size = self.buffer.len * 2;
            const buffer = try self.allocator.alloc(typ, size);
            @memcpy(buffer[0 .. self.size - self.front], self.buffer[self.front..]);
            @memcpy(buffer[self.size - self.front .. self.size], self.buffer[0..self.front]);
            const new = @This(){
                .allocator = self.allocator,
                .buffer = buffer,
                .front = 0,
                .back = self.size,
                .size = self.size,
            };
            self.allocator.free(self.buffer);
            self.* = new;
        }
    };
}

// MAYBE: condvars + .block_recv()
pub fn Channel(typ: type) type {
    return struct {
        const Dq = Deque(typ);
        const Pinned = struct {
            dq: Dq,
            lock: std.Thread.Mutex = .{},
        };
        pinned: *Pinned,

        pub fn init(alloc: std.mem.Allocator) !@This() {
            const dq = try Dq.init(alloc);
            const pinned = try alloc.create(Pinned);
            pinned.* = .{
                .dq = dq,
            };
            return .{
                .pinned = pinned,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.pinned.lock.lock();
            // defer self.pinned.lock.unlock();
            self.pinned.dq.deinit();
            self.pinned.dq.allocator.destroy(self.pinned);
        }

        pub fn send(self: *@This(), val: typ) !void {
            self.pinned.lock.lock();
            defer self.pinned.lock.unlock();
            try self.pinned.dq.push_back(val);
        }

        pub fn try_recv(self: *@This()) ?typ {
            self.pinned.lock.lock();
            defer self.pinned.lock.unlock();
            return self.pinned.dq.pop_front();
        }

        pub fn can_recv(self: *@This()) bool {
            self.pinned.lock.lock();
            defer self.pinned.lock.unlock();
            return self.pinned.dq.peek_front() != null;
        }
    };
}

pub const Fuse = struct {
    fused: std.atomic.Value(bool) = .{ .raw = false },

    pub fn fuse(self: *@This()) bool {
        return self.fused.swap(true, .release);
    }
    pub fn unfuse(self: *@This()) bool {
        const res = self.fused.swap(false, .release);
        return res;
    }
    pub fn check(self: *@This()) bool {
        return self.fused.load(.acquire);
    }
};

pub const FsFuse = struct {
    // - [emcrisostomo/fswatch](https://github.com/emcrisostomo/fswatch?tab=readme-ov-file#libfswatch)
    // - [libfswatch/c/libfswatch.h Reference](http://emcrisostomo.github.io/fswatch/doc/1.17.1/libfswatch.html/libfswatch_8h.html#ae465ef0618fb1dc6d8b70dee68359ea6)
    const c = @cImport({
        @cInclude("libfswatch/c/libfswatch.h");
    });

    const Event = union(enum) {
        All,
        File: []const u8,
    };
    const Chan = Channel(Event);
    const Ctx = struct {
        handle: c.FSW_HANDLE,
        channel: Chan,
        path: [:0]const u8,
    };

    ctx: *Ctx,
    thread: std.Thread,

    pub fn init(path: [:0]const u8) !@This() {
        const rpath = try std.fs.cwd().realpathAlloc(allocator, path);
        defer allocator.free(rpath);
        const pathZ = try allocator.dupeZ(u8, rpath);

        const watch = try start(pathZ);
        return watch;
    }

    pub fn deinit(self: @This()) void {
        _ = c.fsw_stop_monitor(self.ctx.handle);
        _ = c.fsw_destroy_session(self.ctx.handle);

        // OOF: freezes the thread for a while
        // self.thread.join();

        self.ctx.channel.deinit();
        allocator.free(self.ctx.path);
        allocator.destroy(self.ctx);
    }

    pub fn can_recv(self: *@This()) bool {
        return self.ctx.channel.can_recv();
    }

    pub fn try_recv(self: *@This()) ?Event {
        return self.ctx.channel.try_recv();
    }

    pub fn restart(self: *@This(), path: [:0]const u8) !void {
        self.deinit();
        self.* = try init(path);
    }

    fn start(path: [:0]const u8) !@This() {
        const ok = c.fsw_init_library();
        if (ok != c.FSW_OK) {
            return error.CouldNotCreateFsWatcher;
        }

        const ctxt = try allocator.create(Ctx);
        ctxt.* = .{
            .channel = try Chan.init(allocator),
            .handle = null,
            .path = path,
        };

        const Callbacks = struct {
            fn spawn(ctx: *Ctx) !void {
                ctx.handle = c.fsw_init_session(c.filter_include) orelse return error.CouldNotInitFsWatcher;
                var oke = c.fsw_add_path(ctx.handle, ctx.path.ptr);
                if (oke != c.FSW_OK) {
                    return error.PathAdditionFailed;
                }

                oke = c.fsw_set_recursive(ctx.handle, true);
                if (oke != c.FSW_OK) {
                    return error.FswSetFailed;
                }
                oke = c.fsw_set_latency(ctx.handle, 0.2);
                if (oke != c.FSW_OK) {
                    return error.FswSetFailed;
                }
                oke = c.fsw_set_callback(ctx.handle, @ptrCast(&event_callback), ctx);
                if (oke != c.FSW_OK) {
                    return error.FswSetFailed;
                }

                std.debug.print("starting monitor\n", .{});
                oke = c.fsw_start_monitor(ctx.handle);
                if (oke != c.FSW_OK) {
                    return error.CouldNotStartWatcher;
                }
            }
            fn event_callback(events: [*c]const c.fsw_cevent, num: c_uint, ctx: ?*Ctx) callconv(.C) void {
                var flags = c.NoOp;
                // flags |= c.Created;
                flags |= c.Updated;
                // flags |= c.Removed;
                // flags |= c.Renamed;
                // flags |= c.OwnerModified;
                // flags |= c.AttributeModified;
                // flags |= c.MovedFrom;
                // flags |= c.MovedTo;

                for (events[0..@intCast(num)]) |event| {
                    for (event.flags[0..event.flags_num]) |f| {
                        if (flags & @as(c_int, @intCast(f)) == 0) {
                            continue;
                        }
                        const name = c.fsw_get_event_flag_name(f);
                        // std.debug.print("Path: {s}\n", .{event.path});
                        // std.debug.print("Event Type: {s}\n", .{std.mem.span(name)});

                        if (ctx) |cctx| {
                            const stripped = std.fs.path.relative(allocator, cctx.path, std.mem.span(event.path)) catch unreachable;
                            cctx.channel.send(.{ .File = stripped }) catch unreachable;
                        } else {
                            std.debug.print("Error: Event ignored! type: '{s}' path: '{s}'", .{ event.path, name });
                        }
                    }
                }
            }
        };

        const t = try std.Thread.spawn(.{ .allocator = allocator }, Callbacks.spawn, .{ctxt});
        return .{
            .ctx = ctxt,
            .thread = t,
        };
    }
};

pub const ImageMagick = struct {
    // - [ImageMagick – Sitemap](https://imagemagick.org/script/sitemap.php#program-interfaces)
    // - [ImageMagick – MagickWand, C API](https://imagemagick.org/script/magick-wand.php)
    // - [ImageMagick – MagickCore, Low-level C API](https://imagemagick.org/script/magick-core.php)
    const magick = @cImport({
        // @cInclude("MagickCore/MagickCore.h");
        @cDefine("MAGICKCORE_HDRI_ENABLE", "1");
        @cInclude("MagickWand/MagickWand.h");
    });

    pub fn Pixel(typ: type) type {
        return extern struct {
            r: typ,
            g: typ,
            b: typ,
            a: typ,
        };
    }
    pub fn Image(pix: type) type {
        return struct {
            buffer: []pix,
            height: usize,
            width: usize,

            pub fn deinit(self: *@This()) void {
                allocator.free(self.buffer);
            }
        };
    }

    pub const FloatImage = Image(Pixel(f32));
    pub const HalfImage = Image(Pixel(f16));
    pub const UnormImage = Image(Pixel(u8));

    pub fn decode_jpg(bytes: []u8, comptime typ: enum { unorm, half, float }) !(switch (typ) {
        .unorm => Image(Pixel(u8)),
        .half => Image(Pixel(f16)),
        .float => Image(Pixel(f32)),
    }) {
        magick.MagickWandGenesis();
        const wand = magick.NewMagickWand() orelse {
            return error.CouldNotGetWand;
        };
        defer _ = magick.DestroyMagickWand(wand);

        // const pwand = magick.NewPixelWand() orelse {
        //     return error.CouldNotGetWand;
        // };
        // defer _ = magick.DestroyPixelWand(pwand);
        // if (magick.PixelSetColor(pwand, "#28282800") == magick.MagickFalse) {
        //     return error.CouldNotSetPWandColor;
        // }
        // if (magick.MagickSetBackgroundColor(wand, pwand) == magick.MagickFalse) {
        //     return error.CouldNotSetBgColor;
        // }

        if (magick.MagickReadImageBlob(wand, bytes.ptr, bytes.len) == magick.MagickFalse) {
            return error.CouldNotReadImage;
        }

        const img_width = magick.MagickGetImageWidth(wand);
        const img_height = magick.MagickGetImageHeight(wand);

        const buffer = try allocator.alloc(Pixel(switch (typ) {
            .unorm => u8,
            .half => f32,
            .float => f32,
        }), img_width * img_height);
        errdefer allocator.free(buffer);

        if (magick.MagickExportImagePixels(
            wand,
            0,
            0,
            img_width,
            img_height,
            "RGBA",
            switch (typ) {
                .unorm => magick.CharPixel,
                .half => magick.FloatPixel,
                .float => magick.FloatPixel,
            },
            buffer.ptr,
        ) == magick.MagickFalse) {
            return error.CouldNotRenderToBuffer;
        }

        switch (typ) {
            .half => {
                const half = try allocator.alloc(Pixel(f16), img_width * img_height);
                for (buffer, 0..) |v, i| {
                    half[i] = .{
                        .r = @floatCast(v.r),
                        .g = @floatCast(v.g),
                        .b = @floatCast(v.b),
                        .a = @floatCast(v.a),
                    };
                }
                allocator.free(buffer);
                return .{
                    .buffer = half,
                    .height = img_height,
                    .width = img_width,
                };
            },
            else => {
                return .{
                    .buffer = buffer,
                    .height = img_height,
                    .width = img_width,
                };
            },
        }
    }
};

pub const Glslc = struct {
    pub const Compiler = struct {
        pub const Opt = enum {
            none,
            small,
            fast,
        };
        opt: Opt = .none,
        lang: enum {
            glsl,
            hlsl,
        } = .glsl,
        stage: enum {
            vertex,
            fragment,
            compute,
        } = .fragment,
        env: enum {
            vulkan1_3,
            vulkan1_2,
            vulkan1_1,
            vulkan1_0,
        } = .vulkan1_0,
        pub const OutputType = enum {
            assembly,
            spirv,
        };
        pub const Code = union(enum) {
            code: struct {
                src: []const u8,
                definitions: []const []const u8,
            },
            path: struct {
                main: []const u8,
                include: []const []const u8,
                definitions: []const []const u8,
            },
        };

        pub const Err = error{
            GlslcErroredOut,
        };
        pub fn CompileResult(out: OutputType) type {
            return Result(switch (out) {
                .spirv => []u32,
                .assembly => []u8,
            }, Err);
        }

        pub fn dump_assembly(self: @This(), alloc: std.mem.Allocator, code: *const Code) !Result(void, Err) {
            // std.debug.print("{s}\n", .{code});
            const res = try self.compile(alloc, code, .assembly);
            switch (res) {
                .Ok => |bytes| {
                    defer alloc.free(bytes);
                    std.debug.print("{s}\n", .{bytes});
                    return .Ok;
                },
                .Err => |err| {
                    return .{ .Err = err };
                },
            }
        }

        pub fn compile(self: @This(), alloc: std.mem.Allocator, code: *const Code, comptime output_type: OutputType) !CompileResult(output_type) {
            var args = std.ArrayList([]const u8).init(alloc);
            defer {
                for (args.items) |arg| {
                    alloc.free(arg);
                }
                args.deinit();
            }
            try args.append(try alloc.dupe(u8, "glslc"));
            try args.append(try alloc.dupe(u8, switch (self.stage) {
                .fragment => "-fshader-stage=fragment",
                .vertex => "-fshader-stage=vertex",
                .compute => "-fshader-stage=compute",
            }));
            try args.append(try alloc.dupe(u8, switch (self.lang) {
                .glsl => "-xglsl",
                .hlsl => "-xhlsl",
            }));
            try args.append(try alloc.dupe(u8, switch (self.env) {
                .vulkan1_3 => "--target-env=vulkan1.3",
                .vulkan1_2 => "--target-env=vulkan1.2",
                .vulkan1_1 => "--target-env=vulkan1.1",
                .vulkan1_0 => "--target-env=vulkan1.0",
            }));
            try args.append(try alloc.dupe(u8, switch (self.opt) {
                .fast => "-O",
                .small => "-Os",
                .none => "-O0",
            }));
            if (output_type == .assembly) {
                try args.append(try alloc.dupe(u8, "-S"));
            }
            try args.append(try alloc.dupe(u8, "-o-"));
            switch (code.*) {
                .code => |src| {
                    for (src.definitions) |def| {
                        try args.append(try std.fmt.allocPrint(alloc, "-D{s}", .{def}));
                    }
                    try args.append(try alloc.dupe(u8, "-"));
                },
                .path => |paths| {
                    for (paths.definitions) |def| {
                        try args.append(try std.fmt.allocPrint(alloc, "-D{s}", .{def}));
                    }
                    for (paths.include) |inc| {
                        try args.append(try alloc.dupe(u8, "-I"));
                        try args.append(try alloc.dupe(u8, inc));
                    }
                    try args.append(try alloc.dupe(u8, paths.main));
                },
            }

            // for (args.items) |arg| {
            //     std.debug.print("{s} ", .{arg});
            // }
            // std.debug.print("\n", .{});

            var child = std.process.Child.init(args.items, alloc);
            child.stdin_behavior = .Pipe;
            child.stdout_behavior = .Pipe;
            child.stderr_behavior = .Pipe;

            try child.spawn();

            const stdin = child.stdin orelse return error.NoStdin;
            child.stdin = null;
            const stdout = child.stdout orelse return error.NoStdout;
            child.stdout = null;
            const stderr = child.stderr orelse return error.NoStderr;
            child.stderr = null;
            defer stdout.close();
            defer stderr.close();

            switch (code.*) {
                .code => |src| {
                    try stdin.writeAll(src.src);
                },
                .path => {},
            }
            stdin.close();

            // similar to child.collectOutput
            const max_output_bytes = 1000 * 1000;
            var poller = std.io.poll(allocator, enum { stdout, stderr }, .{
                .stdout = stdout,
                .stderr = stderr,
            });
            defer poller.deinit();

            while (try poller.poll()) {
                if (poller.fifo(.stdout).count > max_output_bytes)
                    return error.StdoutStreamTooLong;
                if (poller.fifo(.stderr).count > max_output_bytes)
                    return error.StderrStreamTooLong;
            }

            const err = try child.wait();
            blk: {
                var err_buf = std.ArrayList(u8).init(alloc);

                switch (err) {
                    .Exited => |e| {
                        if (e != 0) {
                            _ = try err_buf.writer().print("exited with code: {}\n", .{e});
                        } else {
                            err_buf.deinit();
                            break :blk;
                        }
                    },
                    // .Signal => |code| {},
                    // .Stopped => |code| {},
                    // .Unknown => |code| {},
                    else => |e| {
                        try err_buf.writer().print("exited with code: {}\n", .{e});
                    },
                }

                const fifo = poller.fifo(.stderr);
                try err_buf.appendSlice(fifo.buf[fifo.head..][0..fifo.count]);
                return .{
                    .Err = .{
                        .err = Err.GlslcErroredOut,
                        .msg = try err_buf.toOwnedSlice(),
                    },
                };
            }

            const fifo = poller.fifo(.stdout);
            var aligned = std.ArrayListAligned(u8, 4).init(allocator);
            try aligned.appendSlice(fifo.buf[fifo.head..][0..fifo.count]);
            const bytes = try aligned.toOwnedSlice();
            return .{ .Ok = switch (output_type) {
                .spirv => std.mem.bytesAsSlice(u32, bytes),
                .assembly => bytes,
            } };
        }
    };
};
