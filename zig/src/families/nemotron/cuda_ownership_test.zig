//! The A6 receipt battery: close and partial-failure ownership for three named surfaces, zero-GPU fakes.

const std = @import("std");
const cuda = @import("cuda");
const head = @import("cuda_lilicorr_head.zig");
const devicez = @import("cuda_lilicorr_device.zig");
const openz = @import("cuda_lilicorr_open.zig");
const stagez = @import("cuda_lilicorr_stage.zig");
const stackz = @import("cuda_lilicorr_stack.zig");
const gatez = @import("cuda_lilicorr_bodygate.zig");
const lc = @import("cuda_lilicorr.zig");
const wts = @import("cuda_lilicorr_weights.zig");

// -- the recorder + ABI fakes: every acquire/free is timestamped and counted --

const Rec = struct {
    var next_addr: u64 = 0x7a000000;
    var alloc_calls: usize = 0;
    var freed: [64]u64 = @splat(0);
    var free_n: usize = 0;
    var unload_calls: usize = 0;
    var hto_calls: usize = 0;
    var fail_hto: ?usize = null; // the k-th HtoD copy refuses
    var fail_alloc: bool = false; // the next cuMemAlloc refuses
    var host_free_n: usize = 0; // ask 3: pinned-host frees counted separately

    fn reset() void {
        next_addr = 0x7a000000;
        alloc_calls = 0;
        freed = @splat(0);
        free_n = 0;
        unload_calls = 0;
        hto_calls = 0;
        fail_hto = null;
        fail_alloc = false;
        host_free_n = 0;
    }

    fn noteFree(p: u64) void {
        if (p == 0) return;
        freed[free_n] = p;
        free_n += 1;
    }

    /// Every device byte acquired by the trial returned, each address exactly once.
    fn expectOnce(base: cuda.memory.Usage) !void {
        const t = std.testing;
        const now = cuda.memory.usage(false);
        try t.expectEqual(base.device, now.device); // bytes freed == bytes acquired (no strand, no double)
        for (freed[0..free_n]) |f| {
            var seen: usize = 0;
            for (freed[0..free_n]) |g| {
                if (f == g) seen += 1;
            }
            try t.expectEqual(@as(usize, 1), seen); // no address was freed twice
        }
    }
};

const Fakes = struct {
    fn memAlloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        Rec.alloc_calls += 1;
        if (Rec.fail_alloc) return cuda.abi.error_not_ready; // quiet through check(): error.NotReady
        Rec.next_addr += 0x40000;
        p.* = Rec.next_addr;
        return cuda.abi.success;
    }

    fn memFree(p: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        Rec.noteFree(p);
        return cuda.abi.success;
    }

    fn htoD(dst: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        _ = dst;
        Rec.hto_calls += 1;
        if (Rec.fail_hto) |k| {
            if (Rec.hto_calls - 1 == k) return cuda.abi.error_not_ready;
        }
        return cuda.abi.success;
    }

    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.success;
    }

    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        Rec.unload_calls += 1;
        return cuda.abi.success;
    }

    fn hostAlloc(p: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
        p.* = @ptrFromInt(@as(usize, 0x7f000000));
        return cuda.abi.success;
    }

    fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
        Rec.host_free_n += 1;
        return cuda.abi.success;
    }
};

fn armDriver(drv: *cuda.Driver) void {
    drv.* = undefined;
    drv.api.cuMemAlloc_v2 = Fakes.memAlloc;
    drv.api.cuMemFree_v2 = Fakes.memFree;
    drv.api.cuMemcpyHtoD_v2 = Fakes.htoD;
    drv.api.cuStreamSynchronize = Fakes.sync;
    drv.api.cuModuleUnload = Fakes.unload;
    drv.api.cuMemHostAlloc = Fakes.hostAlloc;
    drv.api.cuMemFreeHost = Fakes.hostFree;
}

fn modAt(drv: *cuda.Driver, h: u64) cuda.Module {
    var m: cuda.Module = undefined;
    m.d = drv;
    m.handle = @ptrFromInt(h);
    return m;
}

// -- fixtures --

/// A Weights the loader's shape returns: device arena through the fakes, host layers slice on the caller's gpa.
fn bodyWeights(drv: *cuda.Driver, gpa: std.mem.Allocator) !wts.Weights {
    const one: wts.Layer = undefined;
    return .{
        .fc = undefined,
        .hidden_norm = 0,
        .final_norm = 0,
        .layers = try gpa.dupe(wts.Layer, &.{one}),
        .arena = try cuda.DeviceBuffer.alloc(drv, 4096),
    };
}

/// Host head weights with minuscule slices: only lengths matter to loadDevice's staging.
fn tinyHostWeights(gpa: std.mem.Allocator) !head.Weights {
    const v = struct {
        fn fill(g: std.mem.Allocator, n: usize) ![]f32 {
            const s = try g.alloc(f32, n);
            @memset(s, 0);
            return s;
        }
    }.fill;
    const lin: head.Linear = .{ .w = try v(gpa, 4), .b = try v(gpa, 4) };
    const ls = try gpa.alloc(head.Layer, head.n_layers);
    for (ls) |*l| l.* = .{
        .in_proj = try v(gpa, 4),
        .in_bias = try v(gpa, 4),
        .out_proj = try v(gpa, 4),
        .out_bias = try v(gpa, 4),
        .attn_norm = try v(gpa, 4),
        .up = try v(gpa, 4),
        .up_bias = try v(gpa, 4),
        .down = try v(gpa, 4),
        .down_bias = try v(gpa, 4),
        .mlp_norm = try v(gpa, 4),
    };
    return .{
        .token_proj = lin,
        .pass_proj = lin,
        .ctx_proj = lin,
        .feature_norm_w = try v(gpa, 4),
        .feature_norm_b = try v(gpa, 4),
        .feature_up = lin,
        .feature_down = lin,
        .slot_emb = try v(gpa, 4),
        .rank_emb = try v(gpa, 4),
        .out_norm = try v(gpa, 4),
        .anchor_norm = try v(gpa, 4),
        .factor = lin,
        .rel_bias = try v(gpa, 4),
        .same_bias = try v(gpa, 4),
        .out_head = lin,
        .in_head = lin,
        .anchor_out_head = lin,
        .layers = ls,
    };
}

/// The host Weights fixture the head's own leak gate uses (real shapes, small fills).
fn fullHostWeights(gpa: std.mem.Allocator) !head.Weights {
    const v = struct {
        fn fill(g: std.mem.Allocator, n: usize) ![]f32 {
            const s = try g.alloc(f32, n);
            @memset(s, 0.125);
            return s;
        }
    }.fill;
    const lin: head.Linear = .{ .w = try v(gpa, head.hd * head.model_hidden), .b = try v(gpa, head.hd) };
    const ls = try gpa.alloc(head.Layer, head.n_layers);
    for (ls) |*l| l.* = .{
        .in_proj = try v(gpa, 3 * head.hd * head.hd),
        .in_bias = try v(gpa, 3 * head.hd),
        .out_proj = try v(gpa, head.hd * head.hd),
        .out_bias = try v(gpa, head.hd),
        .attn_norm = try v(gpa, head.hd),
        .up = try v(gpa, 2 * head.hd * head.hd),
        .up_bias = try v(gpa, 2 * head.hd),
        .down = try v(gpa, head.hd * 2 * head.hd),
        .down_bias = try v(gpa, head.hd),
        .mlp_norm = try v(gpa, head.hd),
    };
    return .{
        .token_proj = lin,
        .pass_proj = lin,
        .ctx_proj = lin,
        .feature_norm_w = try v(gpa, 5),
        .feature_norm_b = try v(gpa, 5),
        .feature_up = .{ .w = try v(gpa, head.hd * 5), .b = try v(gpa, head.hd) },
        .feature_down = .{ .w = try v(gpa, head.hd * head.hd), .b = try v(gpa, head.hd) },
        .slot_emb = try v(gpa, head.slots * head.hd),
        .rank_emb = try v(gpa, head.topk * head.hd),
        .out_norm = try v(gpa, head.hd),
        .anchor_norm = try v(gpa, head.hd),
        .factor = .{ .w = try v(gpa, head.hd * 3 * head.hd), .b = try v(gpa, head.hd) },
        .rel_bias = try v(gpa, head.heads * (2 * head.slots + 1)),
        .same_bias = try v(gpa, head.heads),
        .out_head = .{ .w = try v(gpa, head.factor * head.hd), .b = try v(gpa, head.factor) },
        .in_head = .{ .w = try v(gpa, head.factor * head.hd), .b = try v(gpa, head.factor) },
        .anchor_out_head = .{ .w = try v(gpa, head.factor * head.hd), .b = try v(gpa, head.factor) },
        .layers = ls,
    };
}

// -- the cells --

test "normal close frees every owned byte once — body weights, staging arena, head arenas (owner A6)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    Rec.reset();
    const base = cuda.memory.usage(false);

    var d: lc.Draft = undefined;
    d.gpa = a;
    d.open_stage = .ready;
    d.ready = true;
    d.ds = .{};
    d.streams = .empty;
    d.pool = .empty;
    d.mod = modAt(&drv, 0x9e100);
    d.mod_q = modAt(&drv, 0x9e200);
    d.tap_ring = try cuda.DeviceBuffer.alloc(&drv, 8);
    var host_head = std.heap.ArenaAllocator.init(a);
    _ = host_head.allocator().alloc(u8, 16) catch unreachable; // chunk FIRST: the copy must see it
    d.head_arena = host_head;
    d.head_loaded = true;
    const dhw: devicez.DeviceWeights = .{
        .token_proj = .{ .w = 0 },
        .pass_proj = .{ .w = 0 },
        .ctx_proj = .{ .w = 0 },
        .feature_norm_w = 0,
        .feature_norm_b = 0,
        .feature_up = .{ .w = 0 },
        .feature_down = .{ .w = 0 },
        .slot_emb = 0,
        .rank_emb = 0,
        .out_norm = 0,
        .anchor_norm = 0,
        .factor = .{ .w = 0 },
        .rel_bias = 0,
        .same_bias = 0,
        .out_head = .{ .w = 0 },
        .in_head = .{ .w = 0 },
        .anchor_out_head = .{ .w = 0 },
        .layers = try a.alloc(devicez.DeviceLayer, 1),
        .arena = try cuda.DeviceBuffer.alloc(&drv, 8192),
    };
    d.dhw = dhw;
    d.w = try bodyWeights(&drv, a);
    inline for (.{ "scratch_buf", "xs_buf", "pm_buf", "pl_buf", "po_buf", "ids_buf", "segments_buf", "cand_ids_buf", "cand_emb_buf", "anchor_buf", "token_states_buf", "draft_picks_buf" }) |name| {
        @field(d, name) = try cuda.DeviceBuffer.alloc(&drv, 16);
    }
    d.staged = try cuda.HostBuffer.alloc(&drv, 1024); // the ask-3 surface: a pinned carve a close must return

    openz.deinit(&d); // the normal close: every owner surface hands back once

    try std.testing.expectEqual(base.device, cuda.memory.usage(false).device); // bytes above and below every stage
    try std.testing.expectEqual(base.host, cuda.memory.usage(false).host); // ask 3: the pinned staging returned too
    try std.testing.expectEqual(@as(usize, 15), Rec.free_n); // 12 scratch + ring + w.arena + dhw.arena (host frees count separately)
    try std.testing.expectEqual(@as(usize, 1), Rec.host_free_n); // ask 3: the pinned staging freed exactly once
    try std.testing.expectEqual(@as(usize, 2), Rec.unload_calls);
    try Rec.expectOnce(base);
}

test "staging arena: a failure at every staging stage frees the arena once, never twice (owner A6)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    var ctx: cuda.Context = undefined;
    ctx.d = &drv;
    ctx.device = 0;
    ctx.handle = null;
    var hw_arena = std.heap.ArenaAllocator.init(a);
    defer hw_arena.deinit();
    const hw = try tinyHostWeights(hw_arena.allocator());
    const base = cuda.memory.usage(false);

    // stage count: uploads per happy run
    Rec.reset();
    var dw = try devicez.loadDevice(a, &ctx, &hw);
    const stages = Rec.hto_calls;
    // happy normal close: the returned DeviceWeights owns both rentals
    dw.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), Rec.free_n); // the arena, once
    try Rec.expectOnce(base);

    // fail at stage k: every byte the call had won by then comes back, nothing twice
    for (0..stages) |k| {
        Rec.reset();
        Rec.fail_hto = k;
        const run_base = cuda.memory.usage(false);
        const r = devicez.loadDevice(a, &ctx, &hw);
        try std.testing.expect(std.meta.isError(r));
        try std.testing.expectEqual(@as(usize, 1), Rec.free_n); // the arena alone dies with the call
        try Rec.expectOnce(run_base);
    }

    // the very first stage: the arena alloc itself refuses — nothing owned, nothing freed
    Rec.reset();
    Rec.fail_alloc = true;
    try std.testing.expectError(error.NotReady, devicez.loadDevice(a, &ctx, &hw));
    try std.testing.expectEqual(@as(usize, 0), Rec.free_n);
    try Rec.expectOnce(cuda.memory.usage(false));
}

test "staging arena: a gpa failure at the layers-slice stage frees the arena (owner A6)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    var ctx: cuda.Context = undefined;
    ctx.d = &drv;
    ctx.device = 0;
    ctx.handle = null;
    var hw_arena = std.heap.ArenaAllocator.init(a);
    defer hw_arena.deinit();
    const hw = try tinyHostWeights(hw_arena.allocator());

    // loadDevice's only host allocation: the layers slice. Fail exactly there.
    var fa = std.testing.FailingAllocator.init(a, .{});
    fa.fail_index = 0;
    Rec.reset();
    const run_base = cuda.memory.usage(false);
    const r = devicez.loadDevice(fa.allocator(), &ctx, &hw);
    try std.testing.expect(std.meta.isError(r));
    try std.testing.expectEqual(@as(usize, 1), Rec.free_n);
    try Rec.expectOnce(run_base);
}

test "body weights: the draft adopts them, then dies before init() — they return whole (owner A6)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    var e_dummy: lc.E = undefined;
    const c: lc.Config = undefined; // init() dies at the checkpoint open before reading it
    var d: lc.Draft = undefined;
    d.open_stage = .none;
    const base = cuda.memory.usage(false);

    Rec.reset();
    const w = try bodyWeights(&drv, a);
    const r = openz.attachWeights(&d, a, std.testing.io, &e_dummy, c, "/nonexistent/a6-no-such-dir", w);
    try std.testing.expect(std.meta.isError(r));
    try std.testing.expectEqual(@as(usize, 1), Rec.free_n); // the body-weights arena, once
    try Rec.expectOnce(base);
    initAteNothing(&d);
}

/// The handover's gate: after a pre-init() death the draft must not hold the weights.
fn initAteNothing(d: *lc.Draft) void {
    if (d.open_stage != .none) {
        // init() cannot have completed: the storage assignment alone never advances the stage
        @panic("adopt must not advance open_stage on its own");
    }
}

test "scratch: a staged-carve upload failure frees the opt-in scratch once and unmarks it (PR 5102 ask 3)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    var ctx: cuda.Context = undefined;
    ctx.d = &drv;
    ctx.device = 0;
    ctx.handle = null;
    var e_dummy: lc.E = undefined;
    e_dummy.ctx = &ctx;
    var hw_arena = std.heap.ArenaAllocator.init(a);
    defer hw_arena.deinit();
    var d: lc.Draft = undefined;
    d.gpa = a;
    d.e = &e_dummy;
    d.ds = .{};
    d.dhw = .{ // non-null: the DeviceWeightsMissing gate passes; only the deinit reads rent the layers+arena
        .token_proj = .{ .w = 0 },
        .pass_proj = .{ .w = 0 },
        .ctx_proj = .{ .w = 0 },
        .feature_norm_w = 0,
        .feature_norm_b = 0,
        .feature_up = .{ .w = 0 },
        .feature_down = .{ .w = 0 },
        .slot_emb = 0,
        .rank_emb = 0,
        .out_norm = 0,
        .anchor_norm = 0,
        .factor = .{ .w = 0 },
        .rel_bias = 0,
        .same_bias = 0,
        .out_head = .{ .w = 0 },
        .in_head = .{ .w = 0 },
        .anchor_out_head = .{ .w = 0 },
        .layers = try a.alloc(devicez.DeviceLayer, 1),
        .arena = try cuda.DeviceBuffer.alloc(&drv, 16),
    };
    d.hw = try fullHostWeights(hw_arena.allocator()); // the uploads read real shape lengths
    defer a.free(d.dhw.?.layers); // the cell's own rentals return with the test
    defer d.dhw.?.arena.free();
    const base = cuda.memory.usage(false);

    Rec.reset();
    Rec.fail_hto = 0; // the FIRST upload refuses: the carve is owned, the unwind must hand it back
    const r = openz.stageDeviceScratch(&d);
    try std.testing.expect(std.meta.isError(r));
    try std.testing.expectEqual(@as(usize, 1), Rec.free_n); // the carve, freed exactly once
    try std.testing.expectEqual(@as(usize, 0), Rec.host_free_n);
    try std.testing.expect(d.ds.staged == false); // ask 3: a freed buf can never stay marked staged
    try std.testing.expectEqual(base.device, cuda.memory.usage(false).device);
}

test "open paths: the real open() hands attachWeights an initialized open_stage (PR 5102 ask 3)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    var e_dummy: lc.E = undefined;
    const c: lc.Config = undefined; // init() dies at the checkpoint open before reading it
    const d = try a.create(lc.Draft); // the REAL handover shape: create() = undefined fields
    defer a.destroy(d);
    d.ready = false; // the three writes real open() performs before attachWeights
    d.ds = .{};
    const base = cuda.memory.usage(false);

    Rec.reset();
    const w = try bodyWeights(&drv, a);
    const r = openz.attachWeights(d, a, std.testing.io, &e_dummy, c, "/nonexistent/a-f3-no-such-dir", w);
    try std.testing.expect(std.meta.isError(r));
    try std.testing.expectEqual(@as(usize, 1), Rec.free_n); // the body-weights arena, once
    try Rec.expectOnce(base);
    if (d.open_stage != .none) @panic("open() must initialize open_stage before init()"); // undefined read REDs pre-cure
}

test "scorer result: a pair-alloc failure frees the first result allocation (owner A6)" {
    var wundef: head.Weights = undefined; // the injected stage fails before any weight is read
    const t = std.testing;

    // stage 0: the first result allocation itself refuses — nothing to strand
    {
        var fa = t.FailingAllocator.init(t.allocator, .{});
        fa.fail_index = 0;
        const r = head.score(fa.allocator(), &wundef, &.{}, &.{}, &.{}, &.{}, false);
        try t.expectError(error.OutOfMemory, r);
    }
    // stage 1: the first allocation succeeded, the second refuses — start must come back
    {
        var fa = t.FailingAllocator.init(t.allocator, .{});
        fa.fail_index = 1;
        const r = head.score(fa.allocator(), &wundef, &.{}, &.{}, &.{}, &.{}, false);
        try t.expectError(error.OutOfMemory, r);
    }
    // the ordinary cleanup stays: a full successful call + the caller's Scores.deinit
    {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const w = try fullHostWeights(arena.allocator());
        const token_emb = try t.allocator.alloc(f32, head.slots * head.topk * head.model_hidden);
        defer t.allocator.free(token_emb);
        @memset(token_emb, 0.125);
        const log_probs = try t.allocator.alloc(f32, head.slots * head.topk);
        defer t.allocator.free(log_probs);
        @memset(log_probs, -1.0);
        const pass_hidden = try t.allocator.alloc(f32, head.model_hidden);
        defer t.allocator.free(pass_hidden);
        @memset(pass_hidden, 0.25);
        const anchor_hidden = try t.allocator.alloc(f32, head.model_hidden);
        defer t.allocator.free(anchor_hidden);
        @memset(anchor_hidden, 0.25);

        const got = try head.score(t.allocator, &w, token_emb, log_probs, pass_hidden, anchor_hidden, true);
        defer got.deinit(t.allocator);
        try t.expectEqual(head.topk, got.start.len);
        try t.expectEqual((head.slots - 1) * head.topk * head.topk, got.pair.len);
    }
}

fn putBlob(io: std.Io, a: std.mem.Allocator, tmp: *std.testing.TmpDir, n: usize, name: []const u8) !void {
    const z = try a.alloc(u8, n);
    defer a.free(z);
    @memset(z, 0);
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = z });
}

/// A dummy DeviceWeights: stageDeviceScratch only null-checks it, so zeros with a live arena suffice.
fn dummyDeviceWeights(drv: *cuda.Driver, a: std.mem.Allocator) !devicez.DeviceWeights {
    var w: devicez.DeviceWeights = undefined;
    inline for (.{ "token_proj", "pass_proj", "ctx_proj", "feature_up", "feature_down", "factor", "out_head", "in_head", "anchor_out_head" }) |name| @field(w, name) = .{ .w = 0 };
    inline for (.{ "feature_norm_w", "feature_norm_b", "slot_emb", "rank_emb", "out_norm", "anchor_norm", "rel_bias", "same_bias" }) |name| @field(w, name) = 0;
    w.layers = try a.alloc(devicez.DeviceLayer, 1);
    w.arena = try cuda.DeviceBuffer.alloc(drv, 8192);
    return w;
}

test "opt-in device scratch: a failed upload frees the carve once and unmarks it (maintainer ask 3)" {
    const a = std.testing.allocator;
    var drv: cuda.Driver = undefined;
    armDriver(&drv);
    var ctx: cuda.Context = undefined;
    ctx.d = &drv;
    ctx.device = 0;
    ctx.handle = null;
    var e_dummy: lc.E = undefined;
    e_dummy.ctx = &ctx;
    var hw_arena = std.heap.ArenaAllocator.init(a);
    defer hw_arena.deinit();
    const dhw = try dummyDeviceWeights(&drv, a);
    Rec.reset();
    const base = cuda.memory.usage(false);

    var d: lc.Draft = undefined;
    d.e = &e_dummy;
    d.gpa = a;
    d.hw = try fullHostWeights(hw_arena.allocator());
    d.dhw = dhw;
    defer a.free(d.dhw.?.layers); // the cell's own rentals return with the test
    defer d.dhw.?.arena.free();
    Rec.fail_hto = 0; // the carve's first upload (the fused arm weights) refuses
    const r = openz.stageDeviceScratch(&d);
    try std.testing.expect(std.meta.isError(r));
    try std.testing.expectEqual(@as(usize, 1), Rec.free_n); // the carve handed back exactly once
    try Rec.expectOnce(base);
    try std.testing.expect(!d.ds.staged); // the flag did not outlive its buffer (pre-cure it stayed true)
}

test "fixture ladders free their host buffers on a mid-read failure (maintainer ask 3)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var d_dummy: lc.Draft = undefined; // the second-read failure leaves before any device op or d deref

    // each ladder's FIRST read finds its full-size blob; the SECOND read's file is absent
    try putBlob(io, a, &tmp, 56 * 4096 * 4, "token_embeddings.f32");
    try putBlob(io, a, &tmp, 56 * 3 * 1024 * 4, "qkv1_pre.f32");
    try putBlob(io, a, &tmp, 56 * 1024 * 4, "out_normed.f32");
    try putBlob(io, a, &tmp, 56 * 1024 * 4, "h_pre.f32");

    // every helper dies at its SECOND read with the FIRST buffer alive; a retained blob fails the cell's leak audit
    try std.testing.expectError(error.FileNotFound, stagez.deviceTokenStatesAB(a, io, &d_dummy, root));
    try std.testing.expectError(error.FileNotFound, stagez.deviceLatticeAttentionAB(a, io, &d_dummy, root));
    try std.testing.expectError(error.FileNotFound, stagez.deviceFactorEdgeAB(a, io, &d_dummy, root));
    try std.testing.expectError(error.FileNotFound, stackz.deviceLayerStackAB(a, io, &d_dummy, root));
    try std.testing.expectError(error.FileNotFound, gatez.deviceFullChainAB(a, io, &d_dummy, root));
}

test "fixture reader: the full-size happy read leaves no host byte behind (maintainer ask 3)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const vals = [_]f32{ 1.5, -2.25, 3.0, 0.125, 4096.0, -0.5 };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "probe.f32", .data = std.mem.sliceAsBytes(&vals) });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const got = try stackz.readFixtureF32(io, a, dir, "probe.f32", vals.len);
    defer a.free(got);
    try std.testing.expectEqualSlices(f32, &vals, got);
}

test "the open hand-over defines the staged mark; the carve windows unmark on unwind (maintainer ask 3, structural)" {
    const t = std.testing;
    const src = @embedFile("cuda_lilicorr.zig");
    const mark = std.mem.indexOf(u8, src, "d.open_stage = .none;") orelse std.math.maxInt(usize);
    try t.expect(std.mem.indexOf(u8, src, "try attachWeights").? > mark); // defined BEFORE the early cleanup reads it
    try t.expect(std.mem.indexOf(u8, src, "d.open_stage = .ring") != null); // the staged ladder still climbs
    const open_src = @embedFile("cuda_lilicorr_open.zig");
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, open_src, "d.ds.staged = false;")); // the two carve windows both unmark on unwind
    try t.expectEqual(@as(usize, 1), std.mem.count(u8, open_src, "d.ds.staged = true;")); // exactly the assignment beside the carve
    try t.expect(std.mem.indexOf(u8, open_src, "d.staged.free();") != null); // the normal close hands the pinned staging back
}
