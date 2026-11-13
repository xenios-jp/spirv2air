// Constant folding of AIR intrinsic calls in simplifyInstructions (fold_cpu, fold_gpu): every case is one intrinsic
// (or int -> float cast) and a table of operand tuples. For each tuple the test knows, from Metal's definition of the
// operation and not from the folding code, whether the result is defined bit for bit (then it must fold) or not
// (count 0, bits past 31, a fraction under a rounding, NaN, infinity, -0.0, out-of-range conversions: it must stay a
// call). fold_cpu checks that and the folded values against a host reference; fold_gpu runs the same calls with
// operands read from a buffer, so Apple's compiler and GPU compute them, and compares every folded tuple bit for bit.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "airkit/airkit.h"
#include "instsimplify.h"

namespace foldcases {

using airbc::Value;

enum class Kind : uint8_t {
  ExtractU, ExtractS, Insert, MinS, MinU, MaxS, MaxU, Round, ConvertU, ConvertS, SIToFP, UIToFP
};

struct Tuple {
  std::vector<uint32_t> bits;  // operands (f32 operands as their bits)
  bool fold;                   // the result is defined bit for bit, so it must fold
  uint32_t value;              // host reference of a folded result (bits)
};

struct Case {
  std::string name;   // intrinsic name, or "sitofp" / "uitofp"
  Kind kind;
  unsigned nargs;
  bool floatArg;      // the operand is f32 (rounding, convert)
  bool floatResult;   // the result is f32 (rounding, int -> float)
  std::vector<Tuple> tuples;
};

inline float asFloat(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t asBits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// A finite f32 that is an integer and not -0.0.
inline bool integral(uint32_t b) {
  if (b == 0x80000000u) return false;
  float f = asFloat(b);
  return std::isfinite(f) && std::trunc(f) == f;
}

inline Tuple reference(Kind k, const std::vector<uint32_t>& a) {
  Tuple t{a, false, 0};
  auto set = [&](uint32_t v) { t.fold = true; t.value = v; };
  switch (k) {
    case Kind::ExtractU:
    case Kind::ExtractS: {
      uint32_t base = a[0], offset = a[1], count = a[2];
      if (count == 0 || offset >= 32 || count > 32 - offset) break;
      uint64_t v = (uint64_t(base) >> offset) & ((1ull << count) - 1);
      if (k == Kind::ExtractS && (v >> (count - 1)) & 1) v |= ~((1ull << count) - 1);
      set(uint32_t(v));
      break;
    }
    case Kind::Insert: {
      uint32_t base = a[0], ins = a[1], offset = a[2], count = a[3];
      if (count == 0 || offset >= 32 || count > 32 - offset) break;
      uint64_t field = ((1ull << count) - 1) << offset;
      set(uint32_t((base & ~field) | ((uint64_t(ins) << offset) & field)));
      break;
    }
    case Kind::MinS: set(uint32_t(std::min(int32_t(a[0]), int32_t(a[1])))); break;
    case Kind::MaxS: set(uint32_t(std::max(int32_t(a[0]), int32_t(a[1])))); break;
    case Kind::MinU: set(std::min(a[0], a[1])); break;
    case Kind::MaxU: set(std::max(a[0], a[1])); break;
    case Kind::Round:
      if (integral(a[0])) set(a[0]);
      break;
    case Kind::ConvertU:
    case Kind::ConvertS: {
      if (a[0] == 0x80000000u) { set(0); break; }
      if (!integral(a[0])) break;
      double v = asFloat(a[0]);
      if (k == Kind::ConvertU ? (v >= 0 && v <= 4294967295.0) : (v >= -2147483648.0 && v <= 2147483647.0))
        set(uint32_t(int64_t(v)));
      break;
    }
    case Kind::SIToFP:
    case Kind::UIToFP: {
      double v = k == Kind::SIToFP ? double(int32_t(a[0])) : double(a[0]);
      if (double(float(v)) == v) set(asBits(float(v)));
      break;
    }
  }
  return t;
}

inline std::vector<Case> cases() {
  std::mt19937 rng(0x5eed);
  auto r32 = [&] { return uint32_t(rng()); };
  std::vector<Case> out;
  auto add = [&](std::string name, Kind k, unsigned n, bool fa, bool fr, const std::vector<std::vector<uint32_t>>& ts) {
    Case c{std::move(name), k, n, fa, fr, {}};
    for (const auto& t : ts) c.tuples.push_back(reference(k, t));
    out.push_back(std::move(c));
  };
  // Bit fields: every (offset, count) on a few bases, including count 0 and fields that run past bit 31.
  std::vector<std::vector<uint32_t>> ex, ins;
  const uint32_t bases[] = {0u, 63u, 0x80000000u, 0xFFFFFFFFu, 0x12345678u, 0x04000000u, 0x0103FC90u};
  for (uint32_t base : bases)
    for (uint32_t offset = 0; offset <= 32; offset += 3)
      for (uint32_t count = 0; count <= 33; count += (count < 8 ? 1 : 5)) ex.push_back({base, offset, count});
  for (int i = 0; i < 300; ++i) ins.push_back({r32(), r32(), r32() % 33, r32() % 34});
  for (uint32_t offset : {0u, 1u, 16u, 31u})
    for (uint32_t count : {0u, 1u, 2u, 31u, 32u}) ins.push_back({0xA5A5A5A5u, 0xFFFFFFFFu, offset, count});
  add("air.extract_bits.u.i32", Kind::ExtractU, 3, false, false, ex);
  add("air.extract_bits.s.i32", Kind::ExtractS, 3, false, false, ex);
  add("air.insert_bits.u.i32", Kind::Insert, 4, false, false, ins);
  std::vector<std::vector<uint32_t>> mm;
  const uint32_t edges[] = {0u, 1u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu, 254u, uint32_t(-254)};
  for (uint32_t x : edges)
    for (uint32_t y : edges) mm.push_back({x, y});
  for (int i = 0; i < 100; ++i) mm.push_back({r32(), r32()});
  add("air.min.s.i32", Kind::MinS, 2, false, false, mm);
  add("air.min.u.i32", Kind::MinU, 2, false, false, mm);
  add("air.max.s.i32", Kind::MaxS, 2, false, false, mm);
  add("air.max.u.i32", Kind::MaxU, 2, false, false, mm);
  // Floats: integers of every magnitude, the limits of the integer ranges, fractions, ties, zeros, denormals,
  // infinities, NaN.
  std::vector<std::vector<uint32_t>> fl;
  const float fv[] = {0.0f, -0.0f, 1.0f, -1.0f, 2.0f, 63.0f, -64.0f, 0.5f, -0.5f, 1.5f, 2.5f, -2.5f, 0.49999997f,
                      8388607.0f, 8388607.5f, 8388608.0f, 16777216.0f, 16777218.0f, 1e9f, 2147483520.0f,
                      2147483648.0f, -2147483648.0f, -2147483904.0f, 4294967040.0f, 4294967296.0f, 1e30f, -1e30f,
                      3.4028235e38f, 1e-40f, -1e-40f, 1.17549435e-38f, INFINITY, -INFINITY, NAN, -NAN};
  for (float f : fv) fl.push_back({asBits(f)});
  for (int i = -40; i <= 40; ++i) fl.push_back({asBits(float(i) * 0.25f)});
  for (int i = 0; i < 64; ++i) fl.push_back({asBits(std::ldexp(float(int32_t(r32()) >> 8), int(r32() % 40) - 20))});
  for (const char* op : {"rint", "trunc", "floor", "ceil"}) {
    add(std::string("air.fast_") + op + ".f32", Kind::Round, 1, true, true, fl);
    add(std::string("air.") + op + ".f32", Kind::Round, 1, true, true, fl);
  }
  add("air.convert.u.i32.f.f32", Kind::ConvertU, 1, true, false, fl);
  add("air.convert.s.i32.f.f32", Kind::ConvertS, 1, true, false, fl);
  std::vector<std::vector<uint32_t>> it;
  const uint32_t iv[] = {0u, 1u, 0xFFFFFFFFu, 16777215u, 16777216u, 16777217u, 16777218u, uint32_t(-16777216),
                         uint32_t(-16777217), 0x7FFFFFFFu, 0x7FFFFF80u, 0x80000000u, 0x80000080u, 0xFFFFFF00u,
                         0xFFFFFF80u, 12345678u, 123456789u};
  for (uint32_t v : iv) it.push_back({v});
  for (int i = 0; i < 64; ++i) it.push_back({r32() >> (r32() % 32)});
  add("sitofp", Kind::SIToFP, 1, false, true, it);
  add("uitofp", Kind::UIToFP, 1, false, true, it);
  return out;
}

// One call (or cast) of `c` on operands `ops`; the result as i32 bits.
inline Value* emit(airkit::Kernel& k, const Case& c, const std::vector<Value*>& ops) {
  airbc::Builder& b = k.builder();
  Value* r;
  if (c.kind == Kind::SIToFP || c.kind == Kind::UIToFP)
    r = b.cast(c.kind == Kind::SIToFP ? airbc::SIToFP : airbc::UIToFP, ops[0], k.f32);
  else
    r = k.callIntrinsic(c.name, c.floatResult ? k.f32 : k.i32, ops, airkit::Kernel::Attrs::Pure);
  return c.floatResult ? b.cast(airbc::BitCast, r, k.i32) : r;
}

// Kernel with every tuple's operands as constants: out[t] = op(tuple t), written by thread 0. Returns the stores.
inline std::vector<airbc::Instruction*> buildConstant(airkit::Kernel& k, const Case& c) {
  unsigned out = k.buffer(k.i32, 0, airkit::Access::ReadWrite, "out");
  k.begin();
  airbc::Builder& b = k.builder();
  std::vector<airbc::Instruction*> stores;
  for (size_t t = 0; t < c.tuples.size(); ++t) {
    std::vector<Value*> ops;
    for (uint32_t bits : c.tuples[t].bits)
      ops.push_back(c.floatArg ? static_cast<Value*>(k.m.constFPBits(k.f32, bits)) : k.m.constInt(k.i32, int32_t(bits)));
    Value* p = b.gep(k.i32, k.arg(out), {k.m.constInt(k.i64, int64_t(t))}, true);
    stores.push_back(b.store(emit(k, c, ops), p, 4));
  }
  b.retVoid();
  return stores;
}

// Kernel with the operands read from a buffer: out[gid] = op(in[gid * nargs ...]); the GPU computes every result.
inline void buildRuntime(airkit::Kernel& k, const Case& c) {
  unsigned in = k.buffer(k.i32, 0, airkit::Access::Read, "in");
  unsigned out = k.buffer(k.i32, 1, airkit::Access::ReadWrite, "out");
  unsigned gid = k.builtin(airkit::Builtin::ThreadPositionInGrid);
  k.begin();
  airbc::Builder& b = k.builder();
  Value* g = b.cast(airbc::ZExt, k.arg(gid), k.i64);
  std::vector<Value*> ops;
  for (unsigned a = 0; a < c.nargs; ++a) {
    Value* idx = b.binop(airbc::Add, b.binop(airbc::Mul, g, k.m.constInt(k.i64, c.nargs)), k.m.constInt(k.i64, a));
    Value* v = b.load(k.i32, b.gep(k.i32, k.arg(in), {idx}, true), 4);
    ops.push_back(c.floatArg ? b.cast(airbc::BitCast, v, k.f32) : v);
  }
  b.store(emit(k, c, ops), b.gep(k.i32, k.arg(out), {g}, true), 4);
  b.retVoid();
}

// Runs simplifyInstructions on a constant kernel and checks every store: a constant (equal to the reference) where
// the tuple must fold, still computed where it must not. Returns the number of wrong tuples; prints the first few.
inline int checkFolds(airkit::Kernel& k, const Case& c, const std::vector<airbc::Instruction*>& stores) {
  spirv2air::simplifyInstructions(k.m, k.function());
  int bad = 0;
  for (size_t t = 0; t < stores.size(); ++t) {
    const Value* v = stores[t]->ops[0];
    const bool folded = v->vk == Value::Const;
    const Tuple& tu = c.tuples[t];
    uint32_t got = folded ? uint32_t(static_cast<const airbc::Constant*>(v)->bits) : 0;
    if (folded != tu.fold || (folded && got != tu.value)) {
      if (bad++ < 4) {
        std::printf("  %s(", c.name.c_str());
        for (size_t a = 0; a < tu.bits.size(); ++a) std::printf("%s0x%08x", a ? ", " : "", tu.bits[a]);
        std::printf("): %s", folded ? "folded" : "not folded");
        if (folded) std::printf(" to 0x%08x", got);
        std::printf(", expected %s", tu.fold ? "folded" : "not folded");
        if (tu.fold) std::printf(" to 0x%08x", tu.value);
        std::printf("\n");
      }
    }
  }
  return bad;
}

}  // namespace foldcases
