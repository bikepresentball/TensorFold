//! The chunk tap-stage battery (PR 5102 ask 7): the tap stage's six-tap multi-chunk + wide-chunk retention faces.
const std = @import("std");
const fwd = @import("cuda_forward.zig");
const chunkz = @import("cuda_forward_chunk.zig");

const D = 8; // hidden: a multiple of eight, the tap kernel's width class

fn f2bf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    const lsb: u32 = (u >> 16) & 1;
    return @truncate((u +% 0x7fff +% lsb) >> 16);
}

fn bf2f(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn bf16add(a: u16, b: u16) u16 {
    return f2bf(bf2f(a) + bf2f(b));
}

// The value planes: deterministic per (layer, position), h and r distinct.

fn hBits(layer: u32, pos: usize) u16 {
    return f2bf(@as(f32, @floatFromInt((@as(usize, layer) * 5 + pos) % 31)) * 0.125);
}

fn rBits(layer: u32, pos: usize) u16 {
    return f2bf(@as(f32, @floatFromInt((@as(usize, layer) * 3 + 2 * pos + 1) % 23)) * 0.25);
}

// The recorded fake: fill32 captures META[0]; draftTap executes tf_nem_tap's contract

// (pos = META[0] + m - row0; the keep guard bounds writes to the newest rows; slot

// pos % ring_rows; per-op bf16(h + r)) and counts the writes per slot.

const Rec = struct {
    h: []u16 = &.{}, // the rows' H plane of the current call
    r: []u16 = &.{},
    hold: []u16 = &.{}, // [taps * ring_rows * D] the bytes each slot last received
    writes: []u32 = &.{}, // [taps * ring_rows] the write count per (tap, slot)
    meta0: i64 = 0,
    ring_rows: usize = 0,
    last_keep: usize = 0,

    fn arm(self: *Rec, a: std.mem.Allocator, taps: usize, ring_rows: usize) !void {
        self.ring_rows = ring_rows;
        self.hold = try a.alloc(u16, taps * ring_rows * D);
        self.writes = try a.alloc(u32, taps * ring_rows);
        @memset(self.hold, 0);
        @memset(self.writes, 0);
        self.meta0 = 0;
        self.last_keep = 0;
    }
};

const FakeOps = struct {
    rec: *Rec,

    pub fn fill32(o: FakeOps, dst: u64, value: u32, words: usize) !void {
        _ = dst;
        try std.testing.expectEqual(@as(usize, 1), words); // the chunk writes exactly META[0]
        o.rec.meta0 = @intCast(value);
    }

    pub fn draftTap(o: FakeOps, h: u64, r: u64, ring: u64, meta: u64, ring_rows: usize, taps: usize, tap: usize, row0: usize, rows: usize, keep_rows: usize, d: usize) !void {
        _ = h;
        _ = r;
        _ = ring;
        _ = meta;
        _ = ring_rows;
        try std.testing.expectEqual(@as(usize, 6), taps); // the six-tap configuration
        try std.testing.expectEqual(@as(usize, D), d);
        try std.testing.expectEqual(@as(usize, 0), row0); // the chunk's own rows: row0 = 0
        const rec = o.rec;
        rec.last_keep = keep_rows;
        var m: usize = 0;
        while (m < rows) : (m += 1) {
            const pos: i64 = rec.meta0 + @as(i64, @intCast(m)) - @as(i64, @intCast(row0));
            // tf_nem_tap's guard, verbatim: the rows beyond the newest keep_rows skip.
            if (keep_rows > 0 and pos < rec.meta0 + @as(i64, @intCast(rows)) - @as(i64, @intCast(keep_rows))) continue;
            const slot: usize = @intCast(@mod(pos, @as(i64, @intCast(rec.ring_rows))));
            rec.writes[tap * rec.ring_rows + slot] += 1;
            const at = (tap * rec.ring_rows + slot) * D;
            for (0..D) |i| rec.hold[at + i] = bf16add(rec.h[m * D + i], rec.r[m * D + i]);
        }
    }
};

// The rig's fake faces: exactly the fields the tap stage reads.

const FakeFwd = struct {
    ops: FakeOps,
    b: struct { p_meta: u64 = 0xF000, delta: u64 = 0xE000 } = .{},
    c: struct { hidden: usize = D } = .{},
    draft_taps: [8]u32 = @splat(0),
    n_draft_taps: usize = 0,
    tap_rows: usize = 0,
    tap_ring: u64 = 0x9A0000,
};

const Rig = struct {
    rec: Rec = .{},
    ff: FakeFwd = undefined,
    a: std.mem.Allocator = undefined,

    fn init(self: *Rig, a: std.mem.Allocator, n_taps: usize, ring_rows: usize) !void {
        self.a = a;
        try self.rec.arm(a, n_taps, ring_rows);
        self.ff = .{ .ops = .{ .rec = &self.rec } };
        self.ff.n_draft_taps = n_taps;
        self.ff.tap_rows = ring_rows;
        for (0..n_taps) |ti| self.ff.draft_taps[ti] = 5; // six taps at layer 5: the last-state face
    }

    fn deinit(self: *Rig) void {
        self.a.free(self.rec.hold);
        self.a.free(self.rec.writes);
    }

    // One production tap-stage call at layer 5: the planes load, the frames ride the rig.
    fn taps(self: *Rig, layer: u32, pos: usize, rows: usize) !void {
        const plane = try self.a.alloc(u16, rows * D * 2);
        defer self.a.free(plane);
        self.rec.h = plane[0 .. rows * D];
        self.rec.r = plane[rows * D ..];
        for (0..rows) |m| for (0..D) |i| {
            self.rec.h[m * D + i] = hBits(layer, pos + m);
            self.rec.r[m * D + i] = rBits(layer, pos + m);
        };
        var w: fwd.Walk = .{ .rows = rows, .pos = pos };
        w.x = 0xAB0000 + layer;
        w.delta = .dense;
        try chunkz.chunkTaps(&self.ff, &w, 5);
    }

    fn slotBytes(self: *const Rig, ti: usize, slot: usize, layer: u32, pos: usize) [D]u16 {
        _ = ti;
        _ = slot;
        var want: [D]u16 = undefined;
        for (0..D) |i| want[i] = bf16add(hBits(layer, pos), rBits(layer, pos));
        _ = self;
        return want;
    }
};

test "a multi-chunk prompt records each of the six requested taps at every covered position" {
    const a = std.testing.allocator;
    var rig: Rig = .{};
    try rig.init(a, 6, 16);
    defer rig.deinit();
    const chunks = [_][2]usize{ .{ 0, 7 }, .{ 7, 9 }, .{ 16, 4 } }; // {pos, rows}: a 20-row prompt
    // every chunk's own rows land in all six taps, byte-exact, at their ring slots

    for (chunks) |ch| {
        try rig.taps(5, ch[0], ch[1]);
        for (ch[0]..ch[0] + ch[1]) |p| {
            const slot = p % rig.rec.ring_rows;
            const want = rig.slotBytes(0, slot, 5, p);
            for (0..6) |ti| {
                const at = (ti * rig.rec.ring_rows + slot) * D;
                for (0..D) |i| try std.testing.expectEqual(want[i], rig.rec.hold[at + i]);
            }
        }
    }
    // the final rotation: slot j holds the NEWEST covered position mapped to it, per tap

    for (0..6) |ti| for (0..16) |slot| {
        const pos: usize = if (slot < 4) 16 + slot else slot; // the newest of {j, j+16}
        try std.testing.expect(rig.rec.writes[ti * 16 + slot] >= 1); // every tap covered every slot
        const at = (ti * 16 + slot) * D;
        const want = rig.slotBytes(ti, slot, 5, pos);
        for (0..D) |i| try std.testing.expectEqual(want[i], rig.rec.hold[at + i]);
    };
}

test "a chunk wider than the ring writes only its newest rows, once per slot (retention)" {
    const a = std.testing.allocator;
    var rig: Rig = .{};
    try rig.init(a, 6, 8);
    defer rig.deinit();
    const pos = 40;
    const rows = 12; // 12 > ring 8: the pre-cure keep-all face wrote every row (multi-writer per slot)
    try rig.taps(5, pos, rows);
    try std.testing.expectEqual(@as(usize, 8), rig.rec.last_keep); // keep_rows = @min(rows, ring_rows)
    for (0..6) |ti| {
        for (0..8) |slot| {
            try std.testing.expectEqual(@as(u32, 1), rig.rec.writes[ti * 8 + slot]); // exactly once: no race face
            // the slot holds the NEWEST row mapped to it: pos 44..51 -> slots 4..7,0..3
            const pos_at: usize = if (slot < 4) 48 + slot else 40 + slot;
            const at = (ti * 8 + slot) * D;
            const want = rig.slotBytes(ti, slot, 5, pos_at);
            for (0..D) |i| try std.testing.expectEqual(want[i], rig.rec.hold[at + i]);
        }
    }
}

test "a narrow chunk keeps writing its whole row set (no retention trim below the ring)" {
    const a = std.testing.allocator;
    var rig: Rig = .{};
    try rig.init(a, 6, 8);
    defer rig.deinit();
    try rig.taps(5, 9, 3);
    try std.testing.expectEqual(@as(usize, 3), rig.rec.last_keep);
    for (0..6) |ti| for (0..3) |m| {
        const slot = (9 + m) % 8; // positions 9..11 land at slots 1..3
        try std.testing.expectEqual(@as(u32, 1), rig.rec.writes[ti * 8 + slot]);
        const at = (ti * 8 + slot) * D;
        const want = rig.slotBytes(ti, slot, 5, 9 + m);
        for (0..D) |i| try std.testing.expectEqual(want[i], rig.rec.hold[at + i]);
    };
}
