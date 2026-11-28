//! Generic wrapper for drafter opening in the nemotron family.
const std = @import("std");
const lanes = @import("lanes");
const native = @import("cuda_native.zig");
const cuda = @import("cuda");
const cuda_lilicorr = @import("cuda_lilicorr.zig");
const cuda_draft_config = @import("cuda_draft_config.zig");

/// The opened drafter structure.
pub const Opened = struct {
    drafter: lanes.drafter.Drafter,
    ctx: *anyopaque,
    close: *const fn (*anyopaque) void,
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    prepare: ?struct { ctx: *anyopaque, run: *const fn (*anyopaque) anyerror!lanes.drafter.Facts, bytes: usize } = null,
};

/// Finish the drafter opening process (no-op for nemotron).
pub fn finishDrafterOpen(p: *anyopaque) void {
    _ = p; // the nemotron Owned carries no startup-room callback to clear (qwen's room lives in its family Lanes)
}

/// Open a drafter for the nemotron family.
pub fn openDrafter(gpa: std.mem.Allocator, io: std.Io, p: *anyopaque, dir: []const u8, bits: u8) !Opened {
    if (bits != 4) return error.DrafterBitsUnsupported;

    // Cast p to *native.Owned
    const own: *native.Owned = @ptrCast(@alignCast(p));

    // Open the LiLiCorr drafter
    const d = try cuda_lilicorr.open(gpa, io, own.e, dir);
    errdefer cuda_lilicorr.deinit(d);

    return .{
        .drafter = cuda_lilicorr.drafter(d),
        .ctx = d,
        .close = closeDrafter,
        .reserve = reserveDrafter,
    };
}

/// Close the drafter.
fn closeDrafter(p: *anyopaque) void {
    const d: *cuda_lilicorr.Draft = @ptrCast(@alignCast(p));
    cuda_lilicorr.deinit(d);
    d.gpa.destroy(d);
}

/// Reserve space for the drafter.
fn reserveDrafter(p: *anyopaque, count: u32) anyerror!void {
    const d: *cuda_lilicorr.Draft = @ptrCast(@alignCast(p));
    for (0..count) |_| {
        if (d.pool.items.len > 0) continue;
        const ctx = try cuda_lilicorr.create(d);
        d.pool.append(d.gpa, ctx) catch |err| { // maintainer ask 9: freeing the context may not
            cuda_lilicorr.drop(d, ctx); // soften the failure - propagate the allocation error
            return err; // so startup cannot report a reservation that did not happen
        };
    }
}

test "reservation: a pool-append allocation failure frees the context and propagates (PR 5102 ask 9)" {
    const t = std.testing;
    const Memo = struct {
        var freed_n: usize = 0;

        fn memAlloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            p.* = 0x7c000000;
            return cuda.abi.success;
        }

        fn memFree(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
            freed_n += 1;
            return cuda.abi.success;
        }

        fn memset8(_: cuda.abi.DevicePtr, _: u8, _: usize) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }

        fn memset8Async(p: cuda.abi.DevicePtr, v: u8, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return memset8(p, v, n);
        }
    };
    var drv: cuda.Driver = undefined;
    drv.api.cuMemAlloc_v2 = Memo.memAlloc;
    drv.api.cuMemFree_v2 = Memo.memFree;
    drv.api.cuMemsetD8_v2 = Memo.memset8;
    drv.api.cuMemsetD8Async = Memo.memset8Async;
    var ctx_dev: cuda.Context = undefined;
    ctx_dev = .{ .d = &drv, .device = 0, .handle = null };
    var e_dummy: cuda_lilicorr.E = undefined;
    e_dummy.ctx = &ctx_dev;
    var d: cuda_lilicorr.Draft = undefined;
    d.e = &e_dummy;
    d.c = .{
        .hidden = 128,
        .intermediate = 128,
        .layers = 1,
        .vocab = 16,
        .heads = 2,
        .kv_heads = 1,
        .head_dim = 32,
        .block = 4,
        .mask = 0,
        .window = 4096,
        .group = 8,
        .rank = 4,
        .topk = 2,
        .eps = 1e-5,
        .theta = 10000,
        .taps = .{ 1, 2, 3, 4, 5, 6, 7, 8 },
        .n_taps = 4,
        .parallel = 0,
        .parallel_declared = false,
    };
    var fa = t.FailingAllocator.init(t.allocator, .{});
    fa.fail_index = 1; // create() eats index 0, the pool append itself refuses at 1
    d.gpa = fa.allocator();
    d.pool = .empty;
    Memo.freed_n = 0;

    const r = reserveDrafter(@ptrCast(&d), 1);
    try t.expectError(error.OutOfMemory, r); // startup must not report a silent reservation
    try t.expectEqual(@as(usize, 0), d.pool.items.len); // nothing entered the pool
    try t.expectEqual(@as(usize, 1), Memo.freed_n); // the context freed exactly once via drop
}
