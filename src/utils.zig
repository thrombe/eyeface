const std = @import("std");
const main = @import("main.zig");
const allocator = main.allocator;

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
        pub const Stage = enum {
            vertex,
            fragment,
            compute,
        };
        opt: Opt = .none,
        lang: enum {
            glsl,
            hlsl,
        } = .glsl,
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

        pub fn dump_assembly(
            self: @This(),
            alloc: std.mem.Allocator,
            code: *const Code,
            stage: Stage,
        ) !Result(void, Err) {
            // std.debug.print("{s}\n", .{code});
            const res = try self.compile(alloc, code, .assembly, stage);
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

        pub fn compile(
            self: @This(),
            alloc: std.mem.Allocator,
            code: *const Code,
            comptime output_type: OutputType,
            stage: Stage,
        ) !CompileResult(output_type) {
            var args = std.ArrayList([]const u8).init(alloc);
            defer {
                for (args.items) |arg| {
                    alloc.free(arg);
                }
                args.deinit();
            }
            try args.append(try alloc.dupe(u8, "glslc"));
            try args.append(try alloc.dupe(u8, switch (stage) {
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
