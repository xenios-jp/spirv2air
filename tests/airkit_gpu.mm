// airkit GPU conformance: every case builds a kernel with airkit (bitcode -> mtlb -> newLibraryWithData, no Xcode
// tools) and the same kernel in MSL (newLibraryWithSource), runs both on identical inputs and compares every
// writable buffer and texture bit for bit. Cases with floating-point math run in fast and strict mode.
// Cases run one at a time on one queue (GPU jobs are serialised; keep this test small).
//
// usage: airkit_gpu [case-name-substring]     AIRKIT_DUMP=1 prints the first mismatches in full
#import <Metal/Metal.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "airkit/airkit.h"

using namespace airkit;
using airbc::Type;
using airbc::Value;

namespace {

struct TexSpec {
  MTLPixelFormat format;
  bool half;
  std::vector<uint8_t> init;
};
struct Case {
  std::string name;
  bool fpModes;  // run in fast and strict mode
  std::function<void(Kernel &)> build;
  std::string msl;
  std::vector<std::vector<uint8_t>> buffers;
  std::vector<size_t> threadgroupMem;
  std::vector<TexSpec> textures;
  bool sampler = false;
  MTLSize grid, group;
};

constexpr int kTex = 64;

template <class T> std::vector<uint8_t> bytesOf(const std::vector<T> &v) {
  std::vector<uint8_t> r(v.size() * sizeof(T));
  memcpy(r.data(), v.data(), r.size());
  return r;
}
std::vector<uint8_t> zeros(size_t n) { return std::vector<uint8_t>(n, 0); }
std::vector<float> ramp(size_t n, float scale, float bias) {
  std::vector<float> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = float(i) * scale + bias;
  return r;
}
uint16_t toHalf(float f) {
  __fp16 h = (__fp16)f;
  uint16_t b;
  memcpy(&b, &h, 2);
  return b;
}
std::vector<uint8_t> halfRamp(size_t n, float scale, float bias) {
  std::vector<uint16_t> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = toHalf(float(i % 97) * scale + bias);
  return bytesOf(r);
}
std::vector<uint8_t> texFloat(float phase) {
  std::vector<float> t(kTex * kTex * 4);
  for (size_t i = 0; i < t.size(); ++i) t[i] = std::sin(float(i) * 0.37f + phase);
  return bytesOf(t);
}
std::vector<uint8_t> texHalf(float phase) {
  std::vector<uint16_t> t(kTex * kTex * 4);
  for (size_t i = 0; i < t.size(); ++i) t[i] = toHalf(std::cos(float(i) * 0.21f + phase));
  return bytesOf(t);
}

// ---------------------------------------------------------------- helpers for building bodies
struct Body {
  Kernel &k;
  airbc::Builder &b;
  airbc::Module &m;
  explicit Body(Kernel &kk) : k(kk), b(kk.builder()), m(kk.m) {}
  Value *c32(int64_t v) { return m.constInt(k.i32, v); }
  Value *c64(int64_t v) { return m.constInt(k.i64, v); }
  Value *cf(float v) { return m.constFloat(v); }
  Value *z64(Value *v) { return b.cast(airbc::ZExt, v, k.i64); }
  Value *at(Type *elem, Value *base, Value *idx32) { return b.gep(elem, base, {z64(idx32)}, true); }
  Value *ld(Type *t, Value *p, uint32_t a) { return b.load(t, p, a); }
  void st(Value *v, Value *p, uint32_t a) { b.store(v, p, a); }
  Value *fadd(Value *x, Value *y) { return b.binop(airbc::FAdd, x, y, k.fmf); }
  Value *fmul(Value *x, Value *y) { return b.binop(airbc::FMul, x, y, k.fmf); }
  Value *add(Value *x, Value *y) { return b.binop(airbc::Add, x, y); }
  Value *mul(Value *x, Value *y) { return b.binop(airbc::Mul, x, y); }
  Value *band(Value *x, Value *y) { return b.binop(airbc::And, x, y); }
  Value *splat4(Value *s) {
    Type *vt = m.vecTy(s->type, 4);
    Value *v = m.constUndef(vt);
    for (int i = 0; i < 4; ++i) v = b.insertElement(v, s, c64(i));
    return v;
  }
};

// ---------------------------------------------------------------- cases
Case threadgroupCase() {
  Case c;
  c.name = "threadgroup";
  c.fpModes = true;
  c.msl = R"(
kernel void k(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
              threadgroup float *tga [[threadgroup(0)]],
              uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float tg[64];
  float v = in[gid];
  tg[tid] = v * 2.0f;
  tga[tid] = v + 1.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  out[gid] = tg[63 - tid] + tga[(tid + 1) & 63];
})";
  c.build = [](Kernel &k) {
    unsigned in = k.buffer(k.f32, 0, Access::Read, "in");
    unsigned out = k.buffer(k.f32, 1, Access::ReadWrite, "out");
    unsigned tga = k.threadgroupBuffer(k.f32, 0, "tga");
    unsigned gid = k.builtin(Builtin::ThreadPositionInGrid);
    unsigned tid = k.builtin(Builtin::ThreadIndexInThreadgroup);
    k.begin();
    Body x(k);
    Type *arr = x.m.arrayTy(k.f32, 64);
    auto *tg = x.m.addGlobal("tg", arr, 3, x.m.constUndef(arr), 4);
    Value *t = k.arg(tid);
    Value *v = x.ld(k.f32, x.at(k.f32, k.arg(in), k.arg(gid)), 4);
    x.st(x.fmul(v, x.cf(2.0f)), x.b.gep(arr, tg, {x.c64(0), x.z64(t)}, true), 4);
    x.st(x.fadd(v, x.cf(1.0f)), x.at(k.f32, k.arg(tga), t), 4);
    k.barrier(MemThreadgroup);
    Value *a = x.ld(k.f32, x.b.gep(arr, tg, {x.c64(0), x.z64(x.b.binop(airbc::Sub, x.c32(63), t))}, true), 4);
    Value *bb = x.ld(k.f32, x.at(k.f32, k.arg(tga), x.band(x.add(t, x.c32(1)), x.c32(63))), 4);
    x.st(x.fadd(a, bb), x.at(k.f32, k.arg(out), k.arg(gid)), 4);
    x.b.retVoid();
  };
  c.buffers = {bytesOf(ramp(256, 0.75f, -40.0f)), zeros(256 * 4)};
  c.threadgroupMem = {64 * 4};
  c.grid = MTLSizeMake(256, 1, 1);
  c.group = MTLSizeMake(64, 1, 1);
  return c;
}

Case simdCase() {
  Case c;
  c.name = "simd";
  c.fpModes = true;
  c.msl = R"(
kernel void k(device const float *in [[buffer(0)]], device float *outf [[buffer(1)]],
              device uint *outu [[buffer(2)]], device half *outh [[buffer(3)]],
              uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
              uint sg [[simdgroup_index_in_threadgroup]]) {
  float v = in[gid];
  uint u = as_type<uint>(v);
  outf[gid * 6 + 0] = simd_shuffle(v, ushort((lane + 5) & 31));
  outf[gid * 6 + 1] = simd_shuffle_xor(v, ushort(3));
  outf[gid * 6 + 2] = simd_shuffle_up(v, ushort(2));
  outf[gid * 6 + 3] = simd_shuffle_down(v, ushort(7));
  outf[gid * 6 + 4] = simd_broadcast(v, ushort(9));
  outf[gid * 6 + 5] = simd_sum(v);
  outu[gid * 4 + 0] = simd_shuffle_xor(u, ushort(4));
  outu[gid * 4 + 1] = as_type<uint>(simd_shuffle_down(as_type<int>(u), ushort(1)));
  ulong bal = (ulong)simd_ballot(v > 0.0f);
  outu[gid * 4 + 2] = uint(bal) ^ uint(bal >> 32);
  outu[gid * 4 + 3] = simd_broadcast_first(u) + lane * 1000u + sg * 100000u;
  outh[gid] = simd_shuffle_xor(half(v), ushort(2));
})";
  c.build = [](Kernel &k) {
    unsigned in = k.buffer(k.f32, 0, Access::Read, "in");
    unsigned outf = k.buffer(k.f32, 1, Access::ReadWrite, "outf");
    unsigned outu = k.buffer(k.i32, 2, Access::ReadWrite, "outu");
    unsigned outh = k.buffer(k.f16, 3, Access::ReadWrite, "outh");
    unsigned gid = k.builtin(Builtin::ThreadPositionInGrid);
    unsigned lane = k.builtin(Builtin::ThreadIndexInSimdgroup);
    unsigned sg = k.builtin(Builtin::SimdgroupIndexInThreadgroup);
    k.begin();
    Body x(k);
    Value *g = k.arg(gid), *l = k.arg(lane);
    Value *v = x.ld(k.f32, x.at(k.f32, k.arg(in), g), 4);
    Value *u = x.b.cast(airbc::BitCast, v, k.i32);
    auto outF = [&](int i, Value *val) {
      x.st(val, x.at(k.f32, k.arg(outf), x.add(x.mul(g, x.c32(6)), x.c32(i))), 4);
    };
    auto outU = [&](int i, Value *val) {
      x.st(val, x.at(k.i32, k.arg(outu), x.add(x.mul(g, x.c32(4)), x.c32(i))), 4);
    };
    outF(0, k.simdShuffle(Shuffle::Index, v, x.band(x.add(l, x.c32(5)), x.c32(31))));
    outF(1, k.simdShuffle(Shuffle::Xor, v, x.c32(3)));
    outF(2, k.simdShuffle(Shuffle::Up, v, x.c32(2)));
    outF(3, k.simdShuffle(Shuffle::Down, v, x.c32(7)));
    outF(4, k.simdShuffle(Shuffle::Broadcast, v, x.c32(9)));
    outF(5, k.simdSum(v));
    outU(0, k.simdShuffle(Shuffle::Xor, u, x.c32(4)));
    outU(1, k.simdShuffle(Shuffle::Down, u, x.c32(1), /*isSigned=*/true));
    Value *bal = k.simdBallot(x.b.fcmp(2 /*ogt*/, v, x.cf(0.0f), k.fmf));
    Value *lo = x.b.cast(airbc::Trunc, bal, k.i32);
    Value *hi = x.b.cast(airbc::Trunc, x.b.binop(airbc::LShr, bal, x.c64(32)), k.i32);
    outU(2, x.b.binop(airbc::Xor, lo, hi));
    Value *bf = k.simdBroadcastFirst(u);
    outU(3, x.add(x.add(bf, x.mul(l, x.c32(1000))), x.mul(k.arg(sg), x.c32(100000))));
    Value *h = k.convert(v, k.f16);
    x.st(k.simdShuffle(Shuffle::Xor, h, x.c32(2)), x.at(k.f16, k.arg(outh), g), 2);
    x.b.retVoid();
  };
  c.buffers = {bytesOf(ramp(128, 0.37f, -20.0f)), zeros(128 * 6 * 4), zeros(128 * 4 * 4), zeros(128 * 2)};
  c.grid = MTLSizeMake(128, 1, 1);
  c.group = MTLSizeMake(64, 1, 1);
  return c;
}

std::vector<uint8_t> texInt(uint32_t seed) {
  std::vector<uint32_t> t(kTex * kTex * 4);
  for (size_t i = 0; i < t.size(); ++i) t[i] = uint32_t(i) * 2654435761u + seed;
  return bytesOf(t);
}

// texture2d<uint> and texture2d<int>: read, write and read_write on RGBA32Uint / RGBA32Sint.
Case integerTextureCase() {
  Case c;
  c.name = "integer_texture";
  c.fpModes = false;
  c.msl = R"(
kernel void k(texture2d<uint, access::read> tu [[texture(0)]], texture2d<int, access::read> ti [[texture(1)]],
              texture2d<uint, access::write> wu [[texture(2)]], texture2d<int, access::read_write> rwi [[texture(3)]],
              uint2 gid [[thread_position_in_grid]]) {
  uint4 a = tu.read(gid);
  int4 b = ti.read(gid);
  wu.write(a + uint4(b), gid);
  rwi.write(rwi.read(gid) - b, gid);
})";
  c.build = [](Kernel &k) {
    unsigned tu = k.texture2d(0, Access::Read, Texel::Uint, "tu");
    unsigned ti = k.texture2d(1, Access::Read, Texel::Int, "ti");
    unsigned wu = k.texture2d(2, Access::Write, Texel::Uint, "wu");
    unsigned rwi = k.texture2d(3, Access::ReadWrite, Texel::Int, "rwi");
    unsigned gid = k.builtin(Builtin::ThreadPositionInGrid, 2);
    k.begin();
    Body x(k);
    Value *g = k.arg(gid);
    Value *a = k.read2d(k.arg(tu), g, Access::Read, Texel::Uint);
    Value *b = k.read2d(k.arg(ti), g, Access::Read, Texel::Int);
    k.write2d(k.arg(wu), g, x.add(a, b), Access::Write, Texel::Uint);
    Value *r = k.read2d(k.arg(rwi), g, Access::ReadWrite, Texel::Int);
    k.write2d(k.arg(rwi), g, x.b.binop(airbc::Sub, r, b), Access::ReadWrite, Texel::Int);
    x.b.retVoid();
  };
  c.textures = {{MTLPixelFormatRGBA32Uint, false, texInt(1)},
                {MTLPixelFormatRGBA32Sint, false, texInt(77)},
                {MTLPixelFormatRGBA32Uint, false, std::vector<uint8_t>(kTex * kTex * 16, 0)},
                {MTLPixelFormatRGBA32Sint, false, texInt(12345)}};
  c.grid = MTLSizeMake(kTex, kTex, 1);
  c.group = MTLSizeMake(8, 8, 1);
  return c;
}

Case textureCase() {
  Case c;
  c.name = "texture";
  c.fpModes = true;
  c.msl = R"(
constexpr sampler s_lin(filter::linear, address::clamp_to_edge, coord::normalized);
constexpr sampler s_near(filter::nearest, address::clamp_to_edge, coord::normalized);
kernel void k(texture2d<float, access::sample> tf [[texture(0)]], texture2d<half, access::sample> th [[texture(1)]],
              texture2d<float, access::read> tr [[texture(2)]], texture2d<float, access::write> tw [[texture(3)]],
              texture2d<half, access::read_write> trw [[texture(4)]], sampler sarg [[sampler(0)]],
              uint2 gid [[thread_position_in_grid]]) {
  float2 uv = (float2(gid) + 0.3f) * 0.015625f;
  float4 a = tf.sample(s_lin, uv);
  float4 b = tf.sample(s_near, uv);
  half4 c = th.sample(s_lin, uv);
  float4 d = tr.read(gid);
  float4 e = tf.sample(sarg, uv, level(0.0f));
  half4 f = trw.read(gid);
  float w = float(tf.get_width() + tf.get_height());
  float4 r = a + b * 2.0f;
  r = r + float4(c);
  r = r + d * 8.0f;
  r = r + e * 16.0f;
  tw.write(r + w, gid);
  trw.write(f * 2.0h, gid);
})";
  c.build = [](Kernel &k) {
    unsigned tf = k.texture2d(0, Access::Sample, false, "tf");
    unsigned th = k.texture2d(1, Access::Sample, true, "th");
    unsigned tr = k.texture2d(2, Access::Read, false, "tr");
    unsigned tw = k.texture2d(3, Access::Write, false, "tw");
    unsigned trw = k.texture2d(4, Access::ReadWrite, true, "trw");
    unsigned sarg = k.sampler(0, "sarg");
    unsigned gid = k.builtin(Builtin::ThreadPositionInGrid, 2);
    k.begin();
    Body x(k);
    airbc::Module &m = x.m;
    Type *v2f = m.vecTy(k.f32, 2), *v4f = m.vecTy(k.f32, 4), *v4h = m.vecTy(k.f16, 4);
    Value *g = k.arg(gid);
    Value *gf = k.convert(g, v2f);
    Value *uv = x.fmul(x.fadd(gf, m.constAggregate(v2f, {m.constFloat(0.3f), m.constFloat(0.3f)})),
                       m.constAggregate(v2f, {m.constFloat(0.015625f), m.constFloat(0.015625f)}));
    Value *lin = k.constSampler(kSamplerLinearClamp), *near = k.constSampler(kSamplerNearestClamp);
    Value *a = k.sample2d(k.arg(tf), lin, uv, false);
    Value *b = k.sample2d(k.arg(tf), near, uv, false);
    Value *cc = k.sample2d(k.arg(th), lin, uv, true);
    Value *d = k.read2d(k.arg(tr), g, Access::Read, false);
    Value *e = k.sample2d(k.arg(tf), k.arg(sarg), uv, false, x.cf(0.0f));
    Value *f = k.read2d(k.arg(trw), g, Access::ReadWrite, true);
    Value *w = k.convert(x.add(k.textureWidth(k.arg(tf)), k.textureHeight(k.arg(tf))), k.f32);
    Value *r = x.fadd(a, x.fmul(b, x.splat4(x.cf(2.0f))));
    r = x.fadd(r, k.convert(cc, v4f));
    r = x.fadd(r, x.fmul(d, x.splat4(x.cf(8.0f))));
    r = x.fadd(r, x.fmul(e, x.splat4(x.cf(16.0f))));
    k.write2d(k.arg(tw), g, x.fadd(r, x.splat4(w)), Access::Write);
    Value *two = m.constFPBits(k.f16, 0x4000);
    k.write2d(k.arg(trw), g, x.fmul(f, m.constAggregate(v4h, {static_cast<airbc::Constant *>(two),
                                                               static_cast<airbc::Constant *>(two),
                                                               static_cast<airbc::Constant *>(two),
                                                               static_cast<airbc::Constant *>(two)})),
              Access::ReadWrite);
    x.b.retVoid();
  };
  c.textures = {{MTLPixelFormatRGBA32Float, false, texFloat(0.0f)},
                {MTLPixelFormatRGBA16Float, true, texHalf(0.5f)},
                {MTLPixelFormatRGBA32Float, false, texFloat(1.0f)},
                {MTLPixelFormatRGBA32Float, false, std::vector<uint8_t>(kTex * kTex * 16, 0)},
                {MTLPixelFormatRGBA16Float, true, texHalf(2.0f)}};
  c.sampler = true;
  c.grid = MTLSizeMake(kTex, kTex, 1);
  c.group = MTLSizeMake(8, 8, 1);
  return c;
}

Case simdMatrixCase() {
  Case c;
  c.name = "simdgroup_matrix";
  c.fpModes = true;
  c.msl = R"(
kernel void k(device const half *A [[buffer(0)]], device const half *B [[buffer(1)]],
              device float *C [[buffer(2)]], device const float *F [[buffer(3)]],
              uint lane [[thread_index_in_simdgroup]]) {
  threadgroup half tgA[64];
  tgA[lane] = A[lane];
  tgA[lane + 32] = A[lane + 32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 a, at, b;
  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.25f);
  simdgroup_load(a, tgA, 8);
  simdgroup_load(at, tgA, 8, ulong2(0), true);
  simdgroup_load(b, B, 8);
  simdgroup_multiply_accumulate(acc, a, b, acc);
  simdgroup_multiply_accumulate(acc, at, b, acc);
  simdgroup_store(acc, C, 8);
  simdgroup_store(acc, C + 64, 8, ulong2(0), true);
  simdgroup_float8x8 f, g;
  simdgroup_load(f, F, 8);
  simdgroup_multiply(g, f, acc);
  simdgroup_store(g, C + 128, 8);
})";
  c.build = [](Kernel &k) {
    unsigned A = k.buffer(k.f16, 0, Access::Read, "A");
    unsigned B = k.buffer(k.f16, 1, Access::Read, "B");
    unsigned C = k.buffer(k.f32, 2, Access::ReadWrite, "C");
    unsigned F = k.buffer(k.f32, 3, Access::Read, "F");
    unsigned lane = k.builtin(Builtin::ThreadIndexInSimdgroup);
    k.begin();
    Body x(k);
    airbc::Module &m = x.m;
    Type *arr = m.arrayTy(k.f16, 64);
    auto *tgA = m.addGlobal("tgA", arr, 3, m.constUndef(arr), 2);
    Value *l = k.arg(lane);
    for (int off : {0, 32}) {
      Value *i = off ? x.add(l, x.c32(off)) : l;
      x.st(x.ld(k.f16, x.at(k.f16, k.arg(A), i), 2), x.b.gep(arr, tgA, {x.c64(0), x.z64(i)}, true), 2);
    }
    k.barrier(MemThreadgroup);
    Value *tg0 = m.constGEP(arr, tgA, {m.constInt(k.i64, 0), m.constInt(k.i64, 0)}, true);
    Value *acc = k.matFill(k.f32, x.cf(0.25f));
    Value *a = k.matLoad(k.f16, tg0, x.c64(8), false);
    Value *at = k.matLoad(k.f16, tg0, x.c64(8), true);
    Value *b = k.matLoad(k.f16, k.arg(B), x.c64(8), false);
    acc = k.matMulAcc(a, b, acc);
    acc = k.matMulAcc(at, b, acc);
    k.matStore(acc, k.arg(C), x.c64(8), false);
    k.matStore(acc, x.b.gep(k.f32, k.arg(C), {x.c64(64)}, true), x.c64(8), true);
    Value *f = k.matLoad(k.f32, k.arg(F), x.c64(8), false);
    Value *g = k.matMulAcc(f, acc, m.constNull(m.vecTy(k.f32, 64)));
    k.matStore(g, x.b.gep(k.f32, k.arg(C), {x.c64(128)}, true), x.c64(8), false);
    x.b.retVoid();
  };
  c.buffers = {halfRamp(64, 0.125f, -3.0f), halfRamp(64, -0.0625f, 2.0f), zeros(192 * 4),
               bytesOf(ramp(64, 0.03125f, -1.0f))};
  c.grid = MTLSizeMake(32, 1, 1);
  c.group = MTLSizeMake(32, 1, 1);
  return c;
}

Case mathCase() {
  Case c;
  c.name = "half_math";
  c.fpModes = true;
  c.msl = R"(
kernel void k(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
              device uint *packed [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  float x = in[gid];
  half h = half(x);
  half2 hp = half2(h, half(x * 0.5f));
  packed[gid] = as_type<uint>(hp);
  device float *o = out + gid * 14;
  o[0] = float(h);
  o[1] = fast::exp2(x);
  o[2] = precise::exp2(x);
  o[3] = fast::log2(x);
  o[4] = precise::log2(x);
  o[5] = fast::rsqrt(x);
  o[6] = precise::rsqrt(x);
  o[7] = fast::sqrt(x);
  o[8] = precise::sqrt(x);
  o[9] = fast::tanh(x);
  o[10] = precise::tanh(x);
  o[11] = fma(x, x, 1.0f);
  o[12] = float(exp2(h));
  o[13] = 1.0f / x;
})";
  c.build = [](Kernel &k) {
    unsigned in = k.buffer(k.f32, 0, Access::Read, "in");
    unsigned out = k.buffer(k.f32, 1, Access::ReadWrite, "out");
    unsigned packed = k.buffer(k.i32, 2, Access::ReadWrite, "packed");
    unsigned gid = k.builtin(Builtin::ThreadPositionInGrid);
    k.begin();
    Body x(k);
    airbc::Module &m = x.m;
    Value *g = k.arg(gid);
    Value *v = x.ld(k.f32, x.at(k.f32, k.arg(in), g), 4);
    Value *h = k.convert(v, k.f16);
    Value *h2 = k.convert(x.fmul(v, x.cf(0.5f)), k.f16);
    Type *v2h = m.vecTy(k.f16, 2);
    Value *hp = x.b.insertElement(x.b.insertElement(m.constUndef(v2h), h, x.c64(0)), h2, x.c64(1));
    x.st(x.b.cast(airbc::BitCast, hp, k.i32), x.at(k.i32, k.arg(packed), g), 4);
    Value *base = x.mul(g, x.c32(14));
    auto o = [&](int i, Value *val) { x.st(val, x.at(k.f32, k.arg(out), x.add(base, x.c32(i))), 4); };
    o(0, k.convert(h, k.f32));
    const char *ops[] = {"exp2", "log2", "rsqrt", "sqrt", "tanh"};
    for (int i = 0; i < 5; ++i) {
      o(1 + 2 * i, k.math(ops[i], v, true));
      o(2 + 2 * i, k.math(ops[i], v, false));
    }
    o(11, k.fma(v, v, x.cf(1.0f)));
    o(12, k.convert(k.math("exp2", h, false), k.f32));
    o(13, x.b.binop(airbc::FDiv, x.cf(1.0f), v, k.fmf));
    x.b.retVoid();
  };
  c.buffers = {bytesOf(ramp(64, 0.173f, 0.05f)), zeros(64 * 14 * 4), zeros(64 * 4)};
  c.grid = MTLSizeMake(64, 1, 1);
  c.group = MTLSizeMake(64, 1, 1);
  return c;
}

Case atomicCase() {
  Case c;
  c.name = "atomics";
  c.fpModes = false;
  c.msl = R"(
kernel void k(device atomic_uint *cu [[buffer(0)]], device atomic_int *ci [[buffer(1)]],
              device atomic_float *cf [[buffer(2)]], device uint *out [[buffer(3)]],
              uint gid [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
              uint tgid [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint tgc;
  if (tid == 0) atomic_store_explicit(&tgc, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  atomic_fetch_add_explicit(&cu[0], gid, memory_order_relaxed);
  atomic_fetch_or_explicit(&cu[1], 1u << (gid & 31), memory_order_relaxed);
  atomic_fetch_max_explicit(&ci[0], int(gid) - 100, memory_order_relaxed);
  atomic_fetch_min_explicit(&ci[1], int(gid) - 100, memory_order_relaxed);
  atomic_fetch_add_explicit(&cf[0], 0.5f, memory_order_relaxed);
  atomic_fetch_add_explicit(&tgc, 1u, memory_order_relaxed);
  if (gid == 0) {
    uint e1 = 9u;
    bool ok1 = atomic_compare_exchange_weak_explicit(&cu[3], &e1, 1u, memory_order_relaxed, memory_order_relaxed);
    uint e2 = 5u;
    bool ok2 = false;
    for (int i = 0; i < 64 && !ok2; ++i)
      ok2 = atomic_compare_exchange_weak_explicit(&cu[4], &e2, 6u, memory_order_relaxed, memory_order_relaxed);
    out[8] = ok1 ? 1u : 0u;
    out[9] = e1;
    out[10] = ok2 ? 1u : 0u;
    out[11] = e2;
    out[12] = atomic_exchange_explicit(&cu[5], 77u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) out[tgid] = atomic_load_explicit(&tgc, memory_order_relaxed);
})";
  c.build = [](Kernel &k) {
    unsigned cu = k.buffer(k.i32, 0, Access::ReadWrite, "cu");
    unsigned ci = k.buffer(k.i32, 1, Access::ReadWrite, "ci");
    unsigned cf = k.buffer(k.f32, 2, Access::ReadWrite, "cf");
    unsigned out = k.buffer(k.i32, 3, Access::ReadWrite, "out");
    unsigned gid = k.builtin(Builtin::ThreadPositionInGrid);
    unsigned tid = k.builtin(Builtin::ThreadIndexInThreadgroup);
    unsigned tgid = k.builtin(Builtin::ThreadgroupPositionInGrid);
    k.begin();
    Body x(k);
    airbc::Module &m = x.m;
    auto *tgc = m.addGlobal("tgc", k.i32, 3, m.constUndef(k.i32), 4);
    Value *g = k.arg(gid), *t = k.arg(tid);
    auto *store0 = k.block("store0"), *cont = k.block("cont"), *first = k.block("first");
    auto *firstEnd = k.block("first.end"), *report = k.block("report"), *done = k.block("done");
    auto *loop = k.block("loop"), *loopBody = k.block("loop.body");
    x.b.condBr(x.b.icmp(32 /*eq*/, t, x.c32(0)), store0, cont);
    x.b.setInsertBlock(store0);
    k.atomicStore(tgc, x.c32(0));
    x.b.br(cont);
    x.b.setInsertBlock(cont);
    k.barrier(MemThreadgroup);
    auto cuAt = [&](int i) { return x.b.gep(k.i32, k.arg(cu), {x.c64(i)}, true); };
    k.atomicRMW(AtomicOp::Add, cuAt(0), g);
    k.atomicRMW(AtomicOp::Or, cuAt(1), x.b.binop(airbc::Shl, x.c32(1), x.band(g, x.c32(31))));
    Value *gm = x.b.binop(airbc::Sub, g, x.c32(100));
    k.atomicRMW(AtomicOp::Max, k.arg(ci), gm, true);
    k.atomicRMW(AtomicOp::Min, x.b.gep(k.i32, k.arg(ci), {x.c64(1)}, true), gm, true);
    k.atomicRMW(AtomicOp::Add, k.arg(cf), x.cf(0.5f));
    k.atomicRMW(AtomicOp::Add, tgc, x.c32(1));
    x.b.condBr(x.b.icmp(32, g, x.c32(0)), first, report);

    x.b.setInsertBlock(first);
    auto outAt = [&](int i) { return x.b.gep(k.i32, k.arg(out), {x.c64(i)}, true); };
    // compare_exchange_weak: the intrinsic returns the value it observed (and stores it to *expected on failure);
    // success is observed == expected-before (observed on GPU, see airkit.md).
    Value *e1 = x.b.allocaInst(k.i32, 4);
    x.st(x.c32(9), e1, 4);
    Value *ok1 = k.atomicCmpxchgWeak(cuAt(3), e1, x.c32(1));
    Value *e2 = x.b.allocaInst(k.i32, 4);
    x.st(x.c32(5), e2, 4);
    x.b.br(loop);
    x.b.setInsertBlock(loop);
    airbc::Instruction *i = x.b.phi(k.i32, {});
    Value *ok2 = k.atomicCmpxchgWeak(cuAt(4), e2, x.c32(6));
    Value *succeeded = x.b.icmp(32 /*eq*/, ok2, x.c32(5));
    Value *inext = x.add(i, x.c32(1));
    x.b.condBr(succeeded, firstEnd, loopBody);
    x.b.setInsertBlock(loopBody);
    x.b.condBr(x.b.icmp(40 /*slt*/, inext, x.c32(64)), loop, firstEnd);
    i->ops = {x.c32(0), inext};
    i->bbs = {first, loopBody};
    x.b.setInsertBlock(firstEnd);
    airbc::Instruction *okp = x.b.phi(k.i32, {});
    okp->ops = {ok2, ok2};
    okp->bbs = {loop, loopBody};
    auto succ = [&](Value *v, int expectedBefore) {
      return x.b.cast(airbc::ZExt, x.b.icmp(32, v, x.c32(expectedBefore)), k.i32);
    };
    x.st(succ(ok1, 9), outAt(8), 4);
    x.st(x.ld(k.i32, e1, 4), outAt(9), 4);
    x.st(succ(okp, 5), outAt(10), 4);
    x.st(x.ld(k.i32, e2, 4), outAt(11), 4);
    x.st(k.atomicRMW(AtomicOp::Xchg, cuAt(5), x.c32(77)), outAt(12), 4);
    x.b.br(report);

    x.b.setInsertBlock(report);
    k.barrier(MemThreadgroup);
    Value *tl = k.atomicLoad(tgc);
    auto *write = k.block("write");
    x.b.condBr(x.b.icmp(32, t, x.c32(0)), write, done);
    x.b.setInsertBlock(write);
    x.st(tl, x.at(k.i32, k.arg(out), k.arg(tgid)), 4);
    x.b.br(done);
    x.b.setInsertBlock(done);
    x.b.retVoid();
  };
  std::vector<uint32_t> cu0 = {0, 0, 0, 3, 5, 11};
  std::vector<int32_t> ci0 = {-1000, 1000};
  c.buffers = {bytesOf(cu0), bytesOf(ci0), zeros(4), zeros(16 * 4)};
  c.grid = MTLSizeMake(256, 1, 1);
  c.group = MTLSizeMake(64, 1, 1);
  return c;
}

// ---------------------------------------------------------------- runner
struct Run {
  std::vector<std::vector<uint8_t>> buffers, textures;
};

bool runLibrary(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib, const Case &c, Run &out,
                std::string &err) {
  NSError *e = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:@"k"];
  if (!fn) {
    err = "no function k";
    return false;
  }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&e];
  if (!pso) {
    err = std::string("pipeline: ") + e.localizedDescription.UTF8String;
    return false;
  }
  NSMutableArray<id<MTLBuffer>> *bufs = [NSMutableArray array];
  for (const auto &b : c.buffers)
    [bufs addObject:[dev newBufferWithBytes:b.data() length:b.size() options:MTLResourceStorageModeShared]];
  NSMutableArray<id<MTLTexture>> *texs = [NSMutableArray array];
  for (const TexSpec &t : c.textures) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.format
                                                                                 width:kTex
                                                                                height:kTex
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    d.storageMode = MTLStorageModeShared;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:d];
    [tex replaceRegion:MTLRegionMake2D(0, 0, kTex, kTex)
           mipmapLevel:0
             withBytes:t.init.data()
           bytesPerRow:t.init.size() / kTex];
    [texs addObject:tex];
  }
  id<MTLSamplerState> smp = nil;
  if (c.sampler) {
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new];
    sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear;
    sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
    smp = [dev newSamplerStateWithDescriptor:sd];
  }
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];
  for (NSUInteger i = 0; i < bufs.count; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
  for (NSUInteger i = 0; i < texs.count; ++i) [enc setTexture:texs[i] atIndex:i];
  if (smp) [enc setSamplerState:smp atIndex:0];
  for (size_t i = 0; i < c.threadgroupMem.size(); ++i) [enc setThreadgroupMemoryLength:c.threadgroupMem[i] atIndex:i];
  [enc dispatchThreads:c.grid threadsPerThreadgroup:c.group];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) {
    err = std::string("gpu: ") + cb.error.localizedDescription.UTF8String;
    return false;
  }
  for (id<MTLBuffer> b in bufs) {
    const uint8_t *p = (const uint8_t *)b.contents;
    out.buffers.emplace_back(p, p + b.length);
  }
  for (NSUInteger i = 0; i < texs.count; ++i) {
    std::vector<uint8_t> px(c.textures[i].init.size());
    [texs[i] getBytes:px.data()
          bytesPerRow:px.size() / kTex
           fromRegion:MTLRegionMake2D(0, 0, kTex, kTex)
          mipmapLevel:0];
    out.textures.push_back(std::move(px));
  }
  return true;
}

int compare(const char *what, size_t index, const std::vector<uint8_t> &got, const std::vector<uint8_t> &want) {
  size_t words = got.size() / 4, bad = 0, first = SIZE_MAX;
  for (size_t w = 0; w < words; ++w)
    if (memcmp(&got[w * 4], &want[w * 4], 4)) {
      if (first == SIZE_MAX) first = w;
      ++bad;
    }
  if (!bad) return 0;
  uint32_t g, wv;
  memcpy(&g, &got[first * 4], 4);
  memcpy(&wv, &want[first * 4], 4);
  float gf, wf;
  memcpy(&gf, &g, 4);
  memcpy(&wf, &wv, 4);
  printf("    %s %zu: %zu/%zu words differ; first [%zu] airkit 0x%08x (%g) msl 0x%08x (%g)\n", what, index, bad,
         words, first, g, gf, wv, wf);
  if (getenv("AIRKIT_DUMP"))
    for (size_t w = first, n = 0; w < words && n < 16; ++w)
      if (memcmp(&got[w * 4], &want[w * 4], 4)) {
        memcpy(&g, &got[w * 4], 4);
        memcpy(&wv, &want[w * 4], 4);
        printf("      [%zu] 0x%08x 0x%08x\n", w, g, wv);
        ++n;
      }
  return 1;
}

}  // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    const char *filter = argc > 1 ? argv[1] : "";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    std::vector<Case> cases = {threadgroupCase(), simdCase(), textureCase(), integerTextureCase(),
                               simdMatrixCase(),  mathCase(), atomicCase()};
    int failures = 0;
    for (const Case &c : cases) {
      if (!strstr(c.name.c_str(), filter)) continue;
      for (int fast = 1; fast >= (c.fpModes ? 0 : 1); --fast) {
        std::string label = c.name + (c.fpModes ? (fast ? " (fast)" : " (strict)") : "");
        Options opts;
        opts.sourceName = c.name;
        opts.fastMath = fast;
        Kernel k("k", opts);
        c.build(k);
        std::vector<uint8_t> lib = buildMetallib({{"k", k.bitcode()}});
        if (getenv("AIRKIT_SAVE")) {
          std::string p = std::string(getenv("AIRKIT_SAVE")) + "/" + c.name + (fast ? ".fast" : ".strict");
          FILE *f = fopen((p + ".metallib").c_str(), "wb");
          fwrite(lib.data(), 1, lib.size(), f);
          fclose(f);
        }
        NSError *e = nil;
        dispatch_data_t data = dispatch_data_create(lib.data(), lib.size(), nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> ours = [dev newLibraryWithData:data error:&e];
        if (!ours) {
          printf("FAIL %s: newLibraryWithData: %s\n", label.c_str(), e.localizedDescription.UTF8String);
          ++failures;
          continue;
        }
        MTLCompileOptions *co = [MTLCompileOptions new];
        co.languageVersion = MTLLanguageVersion3_1;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        co.fastMathEnabled = fast;
#pragma clang diagnostic pop
        NSString *src = [NSString stringWithFormat:@"#include <metal_stdlib>\nusing namespace metal;\n%s",
                                                   c.msl.c_str()];
        id<MTLLibrary> ref = [dev newLibraryWithSource:src options:co error:&e];
        if (!ref) {
          printf("FAIL %s: MSL reference: %s\n", label.c_str(), e.localizedDescription.UTF8String);
          ++failures;
          continue;
        }
        Run a, b;
        std::string err;
        if (!runLibrary(dev, q, ours, c, a, err)) {
          printf("FAIL %s: airkit kernel: %s\n", label.c_str(), err.c_str());
          ++failures;
          continue;
        }
        if (!runLibrary(dev, q, ref, c, b, err)) {
          printf("FAIL %s: MSL kernel: %s\n", label.c_str(), err.c_str());
          ++failures;
          continue;
        }
        int bad = 0;
        for (size_t i = 0; i < a.buffers.size(); ++i) bad += compare("buffer", i, a.buffers[i], b.buffers[i]);
        for (size_t i = 0; i < a.textures.size(); ++i) bad += compare("texture", i, a.textures[i], b.textures[i]);
        printf("%s %s\n", bad ? "FAIL" : "ok  ", label.c_str());
        failures += bad != 0;
      }
    }
    printf(failures ? "%d failing\n" : "all passed\n", failures);
    return failures ? 1 : 0;
  }
}
