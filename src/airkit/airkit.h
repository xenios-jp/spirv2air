// airkit: compute-kernel layer over airbc (AIR bitcode) and mtlb (metallib container).
//
// One Kernel = one AIR module = one metallib function. Declare the arguments, call begin(), emit the body with
// builder() plus the intrinsic helpers below, then bitcode(). buildMetallib() packs any number of kernels into
// a metallib that MTLDevice newLibraryWithData accepts directly (no Xcode tools at runtime).
//
// Target: AIR 2.6, Metal 3.1, macOS 14, metallib file version 2.7 (macOS only; spirv2air::Options selects iOS).
// The helpers describe Metal kernel arguments, intrinsics and module metadata.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "airbc.h"

namespace airkit {

using airbc::Builder;
using airbc::Constant;
using airbc::Function;
using airbc::Type;
using airbc::Value;

enum class Access : uint8_t { Read, ReadWrite, Write, Sample };

// Texel (component) type of a texture argument: texture2d<float>, <half>, <int> or <uint>.
enum class Texel : uint8_t { Float, Half, Int, Uint };

// threadgroup_barrier / simdgroup_barrier mem_flags.
enum MemFlags : uint32_t { MemNone = 0, MemDevice = 1, MemThreadgroup = 2, MemTexture = 4 };

enum class Builtin : uint8_t {
  // uint, uint2 or uint3; MSL requires all of these in one kernel to have the same width.
  ThreadPositionInGrid, ThreadPositionInThreadgroup, ThreadgroupPositionInGrid, ThreadsPerThreadgroup,
  ThreadgroupsPerGrid, ThreadsPerGrid,
  // uint or ushort scalars.
  ThreadIndexInThreadgroup, ThreadIndexInSimdgroup, SimdgroupIndexInThreadgroup, ThreadsPerSimdgroup,
  SimdgroupsPerThreadgroup,
};

enum class Shuffle : uint8_t { Index, Xor, Up, Down, Broadcast };
enum class AtomicOp : uint8_t { Add, Sub, Min, Max, And, Or, Xor, Xchg };

struct Options {
  std::string sourceName = "airkit";
  // Module-level compile options. Per-instruction fast-math flags are separate (Kernel::fmf).
  bool fastMath = true;        // air.compile.fast_math_enable (else _disable)
  bool denormsDisable = true;  // air.compile.denorms_disable (the MSL default)
  uint32_t maxWorkGroupSize = 0;  // kernel "air.max_work_group_size" when non-zero
};

class Kernel {
public:
  explicit Kernel(std::string name, Options opts = {});
  Kernel(const Kernel &) = delete;

  airbc::Module m;
  Type *f16, *f32, *i1, *i8, *i16, *i32, *i64;

  // Fast-math flags applied to floating-point intrinsic calls made through this class (fast-mode MSL puts
  // `fast` on every FP-returning air.* call; strict mode puts none). Defaults from Options::fastMath.
  uint8_t fmf;

  // ---- arguments, in parameter order; each returns its parameter index
  // A device (addrSpace 1) or constant (2) buffer `elem *`. typeName/size/align default from elem
  // (float, half, uint, uchar, ushort, ulong, floatN, ...); struct elements need them explicitly.
  unsigned buffer(Type *elem, unsigned location, Access access, const std::string &name, unsigned addrSpace = 1,
                  std::string typeName = "", uint32_t size = 0, uint32_t align = 0);
  // [[threadgroup(location)]] `threadgroup elem *` argument (size set on the encoder).
  unsigned threadgroupBuffer(Type *elem, unsigned location, const std::string &name, std::string typeName = "",
                             uint32_t size = 0, uint32_t align = 0);
  // width 1..3 for the position/count builtins; for the index builtins width must be 1 (ushort selects i16).
  unsigned builtin(Builtin b, unsigned width = 1, bool ushort = false, const std::string &name = "");
  // texture2d<float|half|int|uint, access> [[texture(location)]]
  unsigned texture2d(unsigned location, Access access, Texel texel, const std::string& name);
  unsigned texture2d(unsigned location, Access access, bool halfTexels, const std::string& name) {
    return texture2d(location, access, halfTexels ? Texel::Half : Texel::Float, name);
  }
  unsigned sampler(unsigned location, const std::string &name);

  // ---- body
  Builder &begin();  // creates the function and its entry block; call after the arguments
  Builder &builder() { return *b_; }
  Function *function() const { return fn_; }
  Value *arg(unsigned i) const;
  airbc::BasicBlock *block(const std::string &name = "");  // a new block in the kernel

  // ---- synchronisation
  void barrier(uint32_t memFlags);           // threadgroup_barrier
  void simdgroupBarrier(uint32_t memFlags);  // simdgroup_barrier

  // ---- SIMD-group functions (lane/delta is converted to i16 as AIR expects)
  Value *simdShuffle(Shuffle op, Value *v, Value *laneOrDelta, bool isSigned = false);
  Value *simdBallot(Value *pred);  // -> i64
  Value *simdBroadcastFirst(Value *v, bool isSigned = false);
  Value *simdSum(Value *v, bool isSigned = false);

  // ---- math. op: exp2 log2 rsqrt sqrt tanh sin cos ... (air.<op>.<type>); fastVariant selects air.fast_<op>
  // (f32 only, as MSL's fast:: namespace). fma/fmin/fmax via call2/call3.
  Value *math(const std::string &op, Value *x, bool fastVariant);
  Value *math2(const std::string &op, Value *x, Value *y, bool fastVariant);
  Value *fma(Value *a, Value *b, Value *c);
  // Numeric conversion: int<->float through air.convert, float<->float with fptrunc/fpext,
  // int<->int with trunc/zext/sext.
  Value *convert(Value *v, Type *to, bool srcSigned = false, bool dstSigned = false);

  // ---- textures
  // A constexpr sampler: `state` is the 64-bit encoded sampler descriptor (see kSampler* below).
  Constant *constSampler(uint64_t state);
  // sample(s, uv) or sample(s, uv, level(lod)) -> <4 x float|half>
  Value *sample2d(Value *tex, Value *smp, Value *uv, bool halfTexels, Value *lod = nullptr);
  // read(uint2 coord[, lod]) -> <4 x float|half>; access is the texture's declared access
  // read2d returns the texel vector: float4, half4, or int4 / uint4 (both <4 x i32>).
  Value* read2d(Value* tex, Value* coord, Access access, Texel texel, Value* lod = nullptr);
  Value* read2d(Value* tex, Value* coord, Access access, bool halfTexels, Value* lod = nullptr) {
    return read2d(tex, coord, access, halfTexels ? Texel::Half : Texel::Float, lod);
  }
  // The texel type of an integer write must be given (Int or Uint); float and half follow the value's type.
  void write2d(Value* tex, Value* coord, Value* texel, Access access, Texel kind, Value* lod = nullptr);
  void write2d(Value *tex, Value *coord, Value *texel, Access access, Value *lod = nullptr);
  Value *textureWidth(Value *tex, Value *lod = nullptr);
  Value *textureHeight(Value *tex, Value *lod = nullptr);

  // ---- simdgroup_matrix<T, 8, 8>: one <64 x T> value per matrix; elements are distributed across the lanes
  // in a hardware-defined layout, so they go through load/store.
  Value *matFill(Type *elem, Value *scalar);
  Value *matLoad(Type *elem, Value *ptr, Value *strideElems, bool transpose, Value *origin = nullptr);
  void matStore(Value *mat, Value *ptr, Value *strideElems, bool transpose, Value *origin = nullptr);
  Value *matMulAcc(Value *a, Value *b, Value *c);  // a*b + c; c's element type is the result type

  // ---- atomics (memory_order_relaxed; device pointers addrspace 1, threadgroup 3)
  Value *atomicRMW(AtomicOp op, Value *ptr, Value *val, bool isSigned = false);
  Value *atomicLoad(Value *ptr);
  void atomicStore(Value *ptr, Value *val);
  // compare_exchange_weak. `expected` is a private (alloca) pointer. Returns the value observed at *ptr (also
  // stored to *expected on failure), so success is `result == expected-before` (verified on GPU).
  Value *atomicCmpxchgWeak(Value *ptr, Value *expected, Value *desired);

  // Finalizes the module (kernel metadata etc.) on first call and writes the wrapped bitcode.
  std::vector<uint8_t> bitcode();
  const std::string &name() const { return name_; }

  // Generic intrinsic call with the attribute class of a known family; exposed for anything not wrapped above.
  enum class Attrs : uint8_t { Pure, Convergent, ConvergentRead, ConvergentWrite, TexSample, TexRead, TexWrite,
                               Atomic };
  Value *callIntrinsic(const std::string &name, Type *ret, const std::vector<Value *> &args, Attrs attrs,
                       const std::vector<std::vector<airbc::Attr>> &paramAttrs = {}, bool fpFlags = false);

private:
  struct Arg {
    Type *type;
    std::vector<airbc::Metadata *> md;  // after the leading index
  };
  Function *declare(const std::string &name, Type *ret, const std::vector<Type *> &params, Attrs attrs,
                    const std::vector<std::vector<airbc::Attr>> &paramAttrs);
  std::string mangle(Type *t, bool isSigned, bool intSign) const;
  std::string typeName(Type *t) const;
  airbc::Metadata *S(const std::string &s) { return m.mdString(s); }
  airbc::Metadata *I(int64_t v) { return m.mdValue(m.constInt(i32, v)); }
  Value *toI16(Value *v);
  void finalize();

  std::string name_;
  Options opts_;
  std::vector<Arg> args_;
  Function *fn_ = nullptr;
  std::unique_ptr<Builder> b_;
  std::map<std::string, Function *> decls_;
  std::vector<airbc::GlobalVariable *> samplers_;
  Type *texTy_ = nullptr, *smpTy_ = nullptr;
  bool finalized_ = false;
};

// Sampler states observed from MSL constexpr samplers (clamp_to_edge, normalized coordinates).
constexpr uint64_t kSamplerLinearClamp = 0x807b'ff00'0008'0a49ull;   // filter::linear
constexpr uint64_t kSamplerNearestClamp = 0x807b'ff00'0008'0049ull;  // filter::nearest

struct LibraryEntry {
  std::string name;
  std::vector<uint8_t> bitcode;
};
// Packs kernels into a metallib (SHA-256 per function: the driver keys its shader cache on it).
std::vector<uint8_t> buildMetallib(const std::vector<LibraryEntry> &kernels);

}  // namespace airkit
