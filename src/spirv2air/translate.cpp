// SPIR-V -> AIR for vertex, fragment, tessellation-evaluation and compute shaders.
// See spirv2air.h for the interface. Graphics uses an argument buffer and descriptor heaps; compute uses direct
// resources at the Metal indices in Options, invocation-id arguments and threadgroup globals.
//
// Pass 1 decodes the id tables (types, constants, decorations, module-scope variables). Pass 2 translates the single
// function: every SPIR-V block becomes one LLVM block, phis are created when their block is reached and get their
// incoming values once every block exists. Inputs, outputs and function variables live in allocas, which mem2reg.cpp
// promotes to SSA values at the end; the entry block initializes them and a synthesized exit block packs the outputs
// into the return struct, so OpReturn and OpKill are both a branch to it.
#include <spirv/unified1/GLSL.std.450.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <spirv/unified1/spirv.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "airbc.h"
#include "metallib/metallib.h"
#include "spirv2air/cleanup.h"
#include "spirv2air/cse.h"
#include "spirv2air/instsimplify.h"
#include "spirv2air/jumpthread.h"
#include "spirv2air/mem2reg.h"
#include "spirv2air/scalarize.h"
#include "spirv2air/simplifycfg.h"
#include "spirv2air/spirv2air.h"

namespace spirv2air {
namespace {

// Per-stage wall time, summed over all modules and printed at exit when S2A_STAGE_TIMES is set in the environment.
struct StageTimer {
  static constexpr const char* kNames[7] = {"translate", "promoteAllocas", "scalarize", "cse",
                                            "cleanup",   "writeBitcode",   "mtlb"};
  static inline double totals[7] = {};
  bool on = std::getenv("S2A_STAGE_TIMES") != nullptr;
  std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
  void lap(int stage) {
    if (!on) return;
    static const int registered = std::atexit([] {
      double sum = 0;
      for (double t : totals) sum += t;
      for (int i = 0; i < 7; ++i)
        std::fprintf(stderr, "stage %-15s %10.1f ms %5.1f%%\n", kNames[i], totals[i] / 1000, 100 * totals[i] / sum);
    });
    (void)registered;
    auto now = std::chrono::steady_clock::now();
    totals[stage] += std::chrono::duration<double, std::micro>(now - last).count();
    last = now;
  }
};

using namespace airbc;

struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& what) { throw Unsupported(what); }

constexpr uint32_t kNone = ~0u;
constexpr const char* kDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-"
    "v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-"
    "v1024:1024:1024-n8:16:32";

// LLVM comparison predicates.
enum : uint8_t {
  F_OEQ = 1, F_OGT = 2, F_OGE = 3, F_OLT = 4, F_OLE = 5, F_ONE = 6, F_ORD = 7, F_UNO = 8, F_UEQ = 9,
  F_UGT = 10, F_UGE = 11, F_ULT = 12, F_ULE = 13, F_UNE = 14,
  I_EQ = 32, I_NE = 33, I_UGT = 34, I_UGE = 35, I_ULT = 36, I_ULE = 37, I_SGT = 38, I_SGE = 39, I_SLT = 40,
  I_SLE = 41,
};
// The air.fast_* math intrinsics (Apple's compiler names them for MSL fast::) carry no fast-math flags. LLVM's nnan and
// ninf let the compiler assume NaN and Inf never occur, nsz lets it treat -0 as +0, and SPIR-V grants those only per
// instruction through FPFastMathMode (NotNaN, NotInf, NSZ), never to the intrinsics' operands in general. Apple's
// compiler writes either `fast` (all flags, fast math on) or no flags (fast:: under -fno-fast-math, clean Q22). A
// matrix of fmin/fmax/clamp/saturate/floor/ceil/trunc/rint/fabs/sqrt/rsqrt/exp2/log2/sin/cos over NaN, +-Inf, +-0, +-1,
// +-2.5 and 0.5, for both intrinsic spellings (air.fast_X and air.X) with no flags, with reassoc nsz arcp afn and with
// fast, gave identical results on the GPU tested (fmin/fmax return the non-NaN operand, zero signs preserved), so the
// flags buy nothing measurable; tests/math_gpu.mm keeps the NaN/Inf/-0 behavior covered. No flags is the choice that
// follows SPIR-V semantics for any input.

struct Inst {
  const uint32_t* w;  // w[0] is the opcode word
  uint16_t op, wc;
  uint32_t operand(unsigned i) const { return w[1 + i]; }
};

// A SPIR-V type, with what the lowering needs precomputed.
struct SType {
  uint16_t op = 0;
  uint32_t elem = kNone;   // Vector/Array/RuntimeArray: element; Pointer: pointee; SampledImage: image
  uint32_t count = 0;      // Vector: components; Array: length; Int/Float: width
  uint32_t storage = 0;    // Pointer: storage class
  spv::Dim dim = spv::DimMax;  // Image
  bool arrayed = false;
  bool multisampled = false;      // Image (compute)
  bool isSigned = false;          // Int
  std::vector<uint32_t> members;  // Struct
  Type* llvm = nullptr;    // value type; null for types without one (pointers, images, runtime arrays)
};

struct Decorations {
  uint32_t builtin = kNone, location = kNone, binding = kNone, set = kNone, stride = 0;
  bool invariant = false, centroid = false, flat = false, noperspective = false, sample = false;
  bool bufferBlock = false, nonWritable = false;  // BufferBlock on a struct type; NonWritable on a variable
  bool noContraction = false;
  bool explicitMathMode = false;
};
struct MemberDecorations {
  std::vector<uint32_t> offset, builtin;
  bool nonWritable = false;
};

// Where an OpVariable or access chain points.
struct Pointer {
  enum Kind : uint8_t { Invalid, Private, Buffer, Resource, FunctionConstant } kind = Invalid;
  uint32_t type = kNone;      // SPIR-V pointee type
  Value* ptr = nullptr;       // Private: pointer to type's LLVM type; Buffer: i8 addrspace(N)* base
  Value* offset = nullptr;    // Buffer: i64 byte offset from ptr
  unsigned addrSpace = 0;     // Buffer
  uint32_t variable = kNone;  // the root OpVariable
  // FunctionConstant: the uint4 constant is member 0 of the block; component selects one element (-1: all four), by
  // the constant index fcComponent or, when that is negative, the dynamic index fcIndex (null: all four).
  int fcComponent = -1;
  Value* fcIndex = nullptr;
};

enum class AttrClass { Pure, Bits, Texture, TextureRead, Derivative, Discard, Atomic, Barrier };

class Translator {
 public:
  Translator(const uint32_t* words, size_t count, const Options& options)
      : words_(words), count_(count), options_(options) {
    // Apple's toolchain sets the AIR version and file version from the deployment target (see mtlb::make_target).
    platform_ = options.platform == Platform::IOS            ? mtlb::Platform::IOS
                : options.platform == Platform::IOSSimulator ? mtlb::Platform::IOSSimulator
                                                             : mtlb::Platform::MacOS;
    target_ = mtlb::make_target(platform_, {options.minOsMajor, options.minOsMinor});
    airVersion_ = mtlb::air_version(platform_, target_.os_version);
  }
  std::vector<uint8_t> run();

 private:
  // ---- pass 1
  void decode();
  void declareType(const Inst& in);
  void declareConstant(const Inst& in);
  Type* llvmType(uint32_t type) const;
  std::string varyingTypeName(uint32_t type) const;
  const SType& type(uint32_t id) const;
  uint32_t constU32(uint32_t id) const;
  bool isConstant(uint32_t id) const { return id < consts_.size() && consts_[id]; }
  bool isConstFloat(uint32_t id, float f) const;

  // ---- signature, entry and exit
  void buildSignature();
  void buildComputeSignature();
  void addSampleInputs(uint32_t sampleId, uint32_t sampleMaskIn);
  void buildEntry();
  void buildComputeEntry();
  Value* inputValue(uint32_t var, Type* t);
  void buildExit();
  Value* clipAndCullDistances(const Pointer& p, int clipMember, int cullMember, Type* result);
  void finalizeMetadata();
  unsigned addArg(Type* t, std::vector<Metadata*> md);

  // ---- body
  void translateFunction();
  void translate(const Inst& in);
  Value* val(uint32_t id) const;
  BasicBlock* block(uint32_t label);

  // ---- memory
  Value* bufferAddress(const Pointer& p, Type* elem);
  Value* load(const Pointer& p);
  Value* loadBuffer(uint32_t type, Value* base, unsigned addrSpace, Value* offset);
  void store(const Pointer& p, Value* v);
  Pointer accessChain(const Pointer& base, const uint32_t* idx, unsigned n);
  Value* loadResource(const Pointer& p);
  Value* tlabSlot(uint32_t offset, Type* pointee);
  Value* addOffset(Value* offset, Value* add);

  // ---- arithmetic helpers
  Value* splat(Value* s, unsigned n);
  Value* perComponent(std::vector<Value*> ops, const std::function<Value*(const std::vector<Value*>&)>& f);
  Function* declare(const std::string& name, Type* ret, const std::vector<Type*>& params, AttrClass cls);
  Value* intrinsic(const std::string& name, Type* ret, const std::vector<Value*>& args, AttrClass cls,
                   uint8_t fmf = 0);
  Value* fast(const char* op, std::vector<Value*> args);
  static bool countIsNonZeroConst(const Value* v) {
    if (v->vk != Value::Const || v->type->kind != Type::Int) return false;
    const auto* c = static_cast<const Constant*>(v);
    return c->ck == Constant::Int && c->bits != 0;
  }
  Value* cint(int64_t v) { return m_.constInt(i32_, v); }
  Value* shiftAmount(Value* v);
  Value* integerDivide(Value* x, Value* y, uint16_t op);
  Value* extendedInteger(const Inst& in);
  Value* extInst(const Inst& in);
  Value* findMsb(Value* x, bool sign);
  Value* geometric(const Inst& in);
  Value* splitFloat(const Inst& in, bool frexp);
  Value* packNormalized(const char* name, Value* x);
  Value* sample(const Inst& in);
  Value* fetch(const Inst& in);
  Value* atomic(const Inst& in);
  Value* atomicExtra(const Inst& in);
  Value* imageQuery(const Inst& in);
  void controlBarrier(const Inst& in);
  const ResourceBinding& binding(uint32_t var) const;
  Value* scalarCast(CastOpc opc, Value* v, Type* to);

  const uint32_t* words_;
  size_t count_;
  const Options& options_;
  mtlb::Platform platform_ = mtlb::Platform::MacOS;
  mtlb::Target target_;
  mtlb::Version airVersion_{2, 6};
  std::vector<Inst> insts_;
  std::vector<uint32_t> uses_;  // how many times each id is used as an operand
  std::unordered_map<uint32_t, std::pair<Value*, Value*>> mulOps_;  // contractable OpFMul results
  size_t functionBegin_ = 0, functionEnd_ = 0;

  Module m_;
  std::unique_ptr<Builder> b_, entryB_;
  Value* demoted_ = nullptr;  // private i1 set by OpDemoteToHelperInvocation, created on first use
  Type *i1_, *i8_, *i16_, *i32_, *i64_, *f16_, *f32_, *v4f32_;
  Type *tlabTy_, *resHeapTy_, *smpHeapTy_, *samplerTy_;
  Type *tex2d_ = nullptr, *tex2dMs_ = nullptr;  // compute textures
  std::map<int, Type*> textureTypes_;           // by spv::Dim, plus kArrayedKey for an arrayed image
  static constexpr int kArrayedKey = 0x100;
  Value* arrayLayer(Value* coordinate);

  std::vector<SType> types_;
  std::vector<Decorations> decor_;
  std::map<uint32_t, MemberDecorations> memberDecor_;
  std::vector<Constant*> consts_;
  std::vector<Value*> vals_;
  std::vector<Pointer> ptrs_;
  std::vector<uint32_t> globals_;  // module-scope OpVariables
  std::vector<uint32_t> varType_;  // OpVariable -> its pointer type
  std::map<uint32_t, uint32_t> varInit_;  // module-scope OpVariable -> its initializer
  std::vector<bool> stored_;       // per variable: written by some OpStore (directly or through a chain)
  std::vector<BasicBlock*> blocks_;  // by label id
  std::vector<std::pair<Instruction*, Inst>> phis_;
  std::map<std::string, Function*> decls_;
  std::map<uint32_t, std::pair<Value*, Value*>> sampledImages_;  // id -> (texture, sampler)
  std::map<uint32_t, uint32_t> sampledImageType_;                 // id -> SPIR-V image type
  std::map<uint32_t, uint32_t> loadedImageType_;                  // compute: id of a loaded image -> its type

  spv::ExecutionModel model_ = spv::ExecutionModelMax;
  uint32_t entryId_ = kNone;
  std::vector<uint32_t> interface_;
  bool earlyFragmentTests_ = false;
  const char* depthQualifier_ = "air.any";  // DepthLess / DepthGreater
  // Tessellation evaluation becomes a post-tessellation vertex function.
  enum class PatchDomain { None, Triangle, Quad } patchDomain_ = PatchDomain::None;
  unsigned controlPoints_ = 1;  // length of the per-control-point input; air.patch_control_point
  uint32_t tessCoord_ = kNone, controlPointInput_ = kNone;
  bool tese() const { return patchDomain_ != PatchDomain::None; }
  bool compute() const { return model_ == spv::ExecutionModelGLCompute; }
  uint32_t localSize_[3] = {1, 1, 1};
  uint32_t glslExt_ = kNone;

  struct Output {
    Value* ptr;    // what to load at exit
    Type* type;
    std::vector<Metadata*> md;
    bool position = false;
    int clipFirst = -1;  // a float4 chunk of gl_ClipDistance: elements clipFirst.. of the loaded array
    int cullMember = -1;  // air.clip_distance with cull distances: the member of gl_CullDistance (see buildExit)
  };
  struct InputInit {
    uint32_t variable;
    std::function<Value*()> value;
  };
  std::vector<Type*> argTypes_;
  std::vector<std::vector<Metadata*>> argMD_;
  std::vector<Output> outputs_;
  std::vector<int> outputMember_;  // per output: member/element of its variable to load, or -1
  std::vector<InputInit> inputInits_;
  std::vector<uint32_t> outputVars_;
  int argVertexId_ = -1, argBaseVertex_ = -1, argTlab_ = -1, argResHeap_ = -1, argSmpHeap_ = -1, argPatchId_ = -1,
      argControlPoints_ = -1;
  std::map<uint32_t, int> argOfInput_;  // Input variable -> argument
  bool usesTextures_ = false, usesSamplers_ = false;

  Function* main_ = nullptr;
  Type* retTy_ = nullptr;
  BasicBlock* entry_ = nullptr;
  BasicBlock* exit_ = nullptr;
  uint32_t firstLabel_ = kNone;
  Value* textureIndices_ = nullptr;

  // Function constants (GraphicsBindings::functionConstantSet): binding -> the "air.fc_initializer" global read.
  std::map<uint32_t, GlobalVariable*> fcGlobals_;
  std::vector<std::string> fcNames_;  // storage for mtlb::FunctionConstant::name
  std::vector<mtlb::FunctionConstant> fcEntries_;
  bool isFunctionConstant(uint32_t var) const;
  int storageSlot(const Decorations& d) const {
    for (size_t k = 0; k < options_.graphics.storageBuffers.size(); ++k)
      if (options_.graphics.storageBuffers[k].set == d.set && options_.graphics.storageBuffers[k].binding == d.binding)
        return int(k);
    return -1;
  }
  Value* functionConstant(uint32_t binding);
};

// ---------------------------------------------------------------- pass 1

const SType& Translator::type(uint32_t id) const {
  if (id >= types_.size() || !types_[id].op) fail("not a type: %" + std::to_string(id));
  return types_[id];
}

// "float", "int2", "uint4", ...: the air.arg_type_name of a varying or colour output. The LLVM type has no signedness,
// so it comes from the SPIR-V OpTypeInt. Each stage names the type its own module declares: Apple's compiler accepts
// a signed/unsigned mismatch between a vertex output and a fragment input at one location and passes the bits through
// unchanged (tests/flat_gpu.mm), so no cross-stage reconciliation is needed (or possible: stages translate separately).
std::string Translator::varyingTypeName(uint32_t id) const {
  const SType& st = type(id);
  const SType& scalar = st.op == spv::OpTypeVector ? type(st.elem) : st;
  std::string base = scalar.op == spv::OpTypeFloat ? "float" : scalar.isSigned ? "int" : "uint";
  return st.op == spv::OpTypeVector ? base + std::to_string(st.count) : base;
}

Type* Translator::llvmType(uint32_t id) const {
  Type* t = type(id).llvm;
  if (!t) fail("type %" + std::to_string(id) + " has no value representation");
  return t;
}

bool isFloatVector(const Type* t) {
  return t->kind == Type::Float || (t->kind == Type::Vector && t->elem->kind == Type::Float);
}

bool isIntVector(const Type* t) {
  return t->kind == Type::Int || (t->kind == Type::Vector && t->elem->kind == Type::Int);
}

// airbc interns +0.0 as a Null constant.
bool isFloatBits(const Constant* c, float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if (c->ck == Constant::Null) return bits == 0;
  return c->ck == Constant::FP && c->bits == bits;
}

uint32_t Translator::constU32(uint32_t id) const {
  // airbc interns zero as a Null constant.
  if (!isConstant(id) || consts_[id]->type->kind != Type::Int ||
      (consts_[id]->ck != Constant::Int && consts_[id]->ck != Constant::Null))
    fail("expected an integer constant, got %" + std::to_string(id));
  return uint32_t(consts_[id]->bits);
}

bool Translator::isConstFloat(uint32_t id, float f) const {
  if (!isConstant(id) || consts_[id]->type != f32_) return false;
  return isFloatBits(consts_[id], f);
}

void Translator::decode() {
  if (count_ < 5 || words_[0] != spv::MagicNumber) fail("not SPIR-V");
  if (options_.halfPackingRounding != HalfPackingRounding::NearestEven &&
      options_.halfPackingRounding != HalfPackingRounding::TowardZero)
    fail("unsupported half packing rounding mode");
  uint32_t bound = words_[3];
  types_.resize(bound);
  decor_.resize(bound);
  consts_.resize(bound);
  vals_.resize(bound);
  ptrs_.resize(bound);
  stored_.resize(bound);
  varType_.resize(bound, kNone);
  for (size_t pos = 5; pos < count_;) {
    uint16_t wc = uint16_t(words_[pos] >> 16), op = uint16_t(words_[pos] & 0xFFFF);
    if (!wc || pos + wc > count_) fail("truncated instruction");
    insts_.push_back({words_ + pos, op, wc});
    pos += wc;
  }
  {
    uses_.assign(bound, 0);
    for (const Inst& in : insts_) {
      if (in.op == spv::OpDecorate || in.op == spv::OpMemberDecorate || in.op == spv::OpName) continue;
      for (unsigned k = 1; k < in.wc; ++k)
        if (in.w[k] < bound) ++uses_[in.w[k]];
    }
    for (uint32_t i = 0; i < bound; ++i)
      if (uses_[i]) --uses_[i];  // its own result id
  }

  std::vector<uint32_t> root(bound, kNone);  // access chain -> its root variable
  for (size_t i = 0; i < insts_.size(); ++i) {
    const Inst& in = insts_[i];
    switch (in.op) {
      case spv::OpCapability: {
        uint32_t c = in.operand(0);
        if (c != spv::CapabilityShader && c != spv::CapabilityDenormFlushToZero &&
            c != spv::CapabilityDerivativeControl && c != spv::CapabilityDemoteToHelperInvocation &&
            c != spv::CapabilityImageQuery && c != spv::CapabilityTessellation &&
            c != spv::CapabilitySampleRateShading && c != spv::CapabilityClipDistance &&
            c != spv::CapabilityCullDistance && c != spv::CapabilityFloatControls2 &&
            c != spv::CapabilityFragmentBarycentricKHR)
          fail("capability " + std::to_string(c));
        break;
      }
      case spv::OpExtInstImport:
        if (std::strcmp(reinterpret_cast<const char*>(in.w + 2), "GLSL.std.450")) fail("extended instruction set");
        glslExt_ = in.operand(0);
        break;
      case spv::OpEntryPoint: {
        if (entryId_ != kNone) fail("more than one entry point");
        model_ = spv::ExecutionModel(in.operand(0));
        entryId_ = in.operand(1);
        unsigned w = 3;  // the name: a nul-terminated string; skip to past the word holding the nul
        while (w < in.wc && (in.w[w] & 0xFF000000u) && (in.w[w] & 0xFF0000u) && (in.w[w] & 0xFF00u) &&
               (in.w[w] & 0xFFu))
          ++w;
        for (++w; w < in.wc; ++w) interface_.push_back(in.w[w]);
        break;
      }
      case spv::OpExecutionMode:
        switch (in.operand(1)) {
          case spv::ExecutionModeEarlyFragmentTests: earlyFragmentTests_ = true; break;
          case spv::ExecutionModeTriangles: patchDomain_ = PatchDomain::Triangle; break;
          case spv::ExecutionModeQuads: patchDomain_ = PatchDomain::Quad; break;
          case spv::ExecutionModeLocalSize:
            for (unsigned c = 0; c < 3; ++c) localSize_[c] = in.operand(2 + c);
            break;
          // Spacing and winding are Metal pipeline state; the caller must set them.
          case spv::ExecutionModeSpacingEqual:
          case spv::ExecutionModeSpacingFractionalEven:
          case spv::ExecutionModeSpacingFractionalOdd:
          case spv::ExecutionModeVertexOrderCw:
          case spv::ExecutionModeVertexOrderCcw:
          case spv::ExecutionModeOriginUpperLeft:
          case spv::ExecutionModeDepthReplacing:
            break;
          case spv::ExecutionModeDepthLess:
            depthQualifier_ = "air.less";
            break;
          case spv::ExecutionModeDepthGreater:
            depthQualifier_ = "air.greater";
            break;
          case spv::ExecutionModeDenormFlushToZero: break;
          case spv::ExecutionModeIsolines:
            // Metal's tessellator has triangle and quad domains only.
            fail("execution mode " + std::to_string(in.operand(1)) + " (Isolines: Metal has no isoline domain)");
          default: fail("execution mode " + std::to_string(in.operand(1)));
        }
        break;
      case spv::OpDecorate: {
        Decorations& d = decor_[in.operand(0)];
        switch (in.operand(1)) {
          case spv::DecorationFPFastMathMode:
            if (in.operand(2) != 0) fail("FPFastMathMode: only None is supported");
            // None forbids contracting this operation with its producers.
            // Keep these values out of the optional implicit-fusion path too.
            d.noContraction = true;
            d.explicitMathMode = true;
            break;
          case spv::DecorationBuiltIn: d.builtin = in.operand(2); break;
          case spv::DecorationLocation: d.location = in.operand(2); break;
          case spv::DecorationBinding: d.binding = in.operand(2); break;
          case spv::DecorationDescriptorSet: d.set = in.operand(2); break;
          case spv::DecorationArrayStride: d.stride = in.operand(2); break;
          case spv::DecorationInvariant: d.invariant = true; break;
          case spv::DecorationCentroid: d.centroid = true; break;
          case spv::DecorationFlat: d.flat = true; break;
          case spv::DecorationNoPerspective: d.noperspective = true; break;
          case spv::DecorationSample:
            d.sample = true;
            break;
          case spv::DecorationBufferBlock:
            d.bufferBlock = true;
            break;
          case spv::DecorationNonWritable:
            d.nonWritable = true;
            break;
          case spv::DecorationCoherent: break;  // device atomics are coherent already (the ZPD counter)
          case spv::DecorationBlock:
          case spv::DecorationNonReadable:
          case spv::DecorationNoContraction:  // no `contract` flag is ever set; NoContraction only blocks the fmuladd below
            d.noContraction = true;
            break;
          default: fail("decoration " + std::to_string(in.operand(1)));
        }
        break;
      }
      case spv::OpMemberDecorate: {
        MemberDecorations& d = memberDecor_[in.operand(0)];
        uint32_t member = in.operand(1);
        if (d.offset.size() <= member) {
          d.offset.resize(member + 1, kNone);
          d.builtin.resize(member + 1, kNone);
        }
        switch (in.operand(2)) {
          case spv::DecorationOffset: d.offset[member] = in.operand(3); break;
          case spv::DecorationBuiltIn: d.builtin[member] = in.operand(3); break;
          case spv::DecorationNonWritable: d.nonWritable = true; break;
          case spv::DecorationRestrict:
          case spv::DecorationCoherent:  // the ZPD counter: device-scope atomics are its only accesses
          case spv::DecorationNonReadable: break;
          default: fail("member decoration " + std::to_string(in.operand(2)));
        }
        break;
      }
      case spv::OpTypeVoid: case spv::OpTypeBool: case spv::OpTypeInt: case spv::OpTypeFloat:
      case spv::OpTypeVector: case spv::OpTypeArray: case spv::OpTypeRuntimeArray: case spv::OpTypeStruct:
      case spv::OpTypePointer: case spv::OpTypeFunction: case spv::OpTypeImage: case spv::OpTypeSampler:
      case spv::OpTypeSampledImage:
        declareType(in);
        break;
      case spv::OpConstant: case spv::OpConstantTrue: case spv::OpConstantFalse: case spv::OpConstantComposite:
      case spv::OpConstantNull:
        declareConstant(in);
        break;
      case spv::OpUndef:  // module scope (spirv-opt hoists them there); function scope is translated in place
        if (!functionBegin_) {
          consts_[in.operand(1)] = m_.constUndef(llvmType(in.operand(0)));
          vals_[in.operand(1)] = consts_[in.operand(1)];
        }
        break;
      case spv::OpVariable:
        varType_[in.operand(1)] = in.operand(0);
        if (!functionBegin_) globals_.push_back(in.operand(1));
        if (!functionBegin_ && in.wc > 4) varInit_[in.operand(1)] = in.operand(3);
        break;
      case spv::OpFunction:
        if (functionBegin_) fail("more than one function");
        functionBegin_ = i;
        break;
      case spv::OpFunctionEnd: functionEnd_ = i; break;
      case spv::OpAccessChain:
      case spv::OpInBoundsAccessChain: {
        uint32_t base = in.operand(2);
        root[in.operand(1)] = root[base] != kNone ? root[base] : base;
        break;
      }
      case spv::OpStore: {
        uint32_t p = in.operand(0);
        stored_[root[p] != kNone ? root[p] : p] = true;
        break;
      }
      default: break;
    }
  }
  if (entryId_ == kNone || !functionBegin_ || !functionEnd_) fail("no entry point");
  if (model_ != spv::ExecutionModelVertex && model_ != spv::ExecutionModelFragment &&
      model_ != spv::ExecutionModelTessellationEvaluation && model_ != spv::ExecutionModelGLCompute)
    fail("execution model " + std::to_string(model_));
  if ((model_ == spv::ExecutionModelTessellationEvaluation) != (patchDomain_ != PatchDomain::None))
    fail("tessellation domain");
}

void Translator::declareType(const Inst& in) {
  uint32_t id = in.operand(0);
  SType& t = types_[id];
  t.op = in.op;
  switch (in.op) {
    case spv::OpTypeVoid: t.llvm = m_.voidTy(); break;
    case spv::OpTypeBool: t.llvm = i1_; break;
    case spv::OpTypeInt:
      t.count = in.operand(1);
      t.isSigned = in.operand(2) != 0;
      if (t.count != 32) fail("int width " + std::to_string(t.count));
      t.llvm = i32_;
      break;
    case spv::OpTypeFloat:
      t.count = in.operand(1);
      if (t.count != 32) fail("float width " + std::to_string(t.count));
      t.llvm = f32_;
      break;
    case spv::OpTypeVector:
      t.elem = in.operand(1);
      t.count = in.operand(2);
      t.llvm = m_.vecTy(llvmType(t.elem), t.count);
      break;
    case spv::OpTypeArray:
      t.elem = in.operand(1);
      t.count = constU32(in.operand(2));
      if (Type* e = type(t.elem).llvm) t.llvm = m_.arrayTy(e, t.count);
      break;
    case spv::OpTypeRuntimeArray: t.elem = in.operand(1); break;
    case spv::OpTypeStruct: {
      std::vector<Type*> elems;
      bool representable = true;
      for (unsigned w = 2; w < in.wc; ++w) {
        t.members.push_back(in.w[w]);
        Type* e = type(in.w[w]).llvm;
        representable = representable && e;
        elems.push_back(e);
      }
      if (representable) t.llvm = m_.literalStructTy(elems);
      break;
    }
    case spv::OpTypePointer:
      t.storage = in.operand(1);
      t.elem = in.operand(2);
      break;
    case spv::OpTypeFunction: break;
    case spv::OpTypeImage:
      t.elem = in.operand(1);
      t.dim = spv::Dim(in.operand(2));
      t.arrayed = in.operand(4) != 0;
      if (compute()) {
        // Sampled 2D images, fetched with OpImageFetch.
        t.multisampled = in.operand(5) != 0;
        if (t.dim != spv::Dim2D || t.arrayed || in.operand(6) != 1) fail("compute image other than a sampled 2D one");
        if (type(t.elem).op != spv::OpTypeFloat && type(t.elem).op != spv::OpTypeInt) fail("image sampled type");
        break;
      }
      if (type(t.elem).op != spv::OpTypeFloat) fail("non-float image");
      if (in.operand(5)) fail("multisampled image");
      if (t.dim != spv::Dim2D && t.dim != spv::DimCube && !(t.dim == spv::Dim3D && !t.arrayed))
        fail("image dimension " + std::to_string(t.dim) + (t.arrayed ? " arrayed" : ""));
      break;
    case spv::OpTypeSampler: break;
    case spv::OpTypeSampledImage: t.elem = in.operand(1); break;
  }
}

void Translator::declareConstant(const Inst& in) {
  uint32_t ty = in.operand(0), id = in.operand(1);
  Type* t = llvmType(ty);
  Constant* c = nullptr;
  switch (in.op) {
    case spv::OpConstant:
      c = t == f32_ ? m_.constFPBits(f32_, in.operand(2)) : m_.constInt(t, int64_t(int32_t(in.operand(2))));
      break;
    case spv::OpConstantTrue: c = m_.constInt(i1_, 1); break;
    case spv::OpConstantFalse: c = m_.constInt(i1_, 0); break;
    case spv::OpConstantNull: c = m_.constNull(t); break;
    case spv::OpConstantComposite: {
      std::vector<Constant*> ops;
      for (unsigned w = 3; w < in.wc; ++w) {
        if (!isConstant(in.w[w])) fail("composite constant operand");
        ops.push_back(consts_[in.w[w]]);
      }
      c = m_.constAggregate(t, ops);
      break;
    }
  }
  consts_[id] = c;
  vals_[id] = c;
}

// ---------------------------------------------------------------- signature

unsigned Translator::addArg(Type* t, std::vector<Metadata*> md) {
  argTypes_.push_back(t);
  argMD_.push_back(std::move(md));
  return unsigned(argTypes_.size() - 1);
}


void Translator::buildSignature() {
  if (compute()) return buildComputeSignature();
  auto S = [&](const std::string& s) { return m_.mdString(s); };
  auto I = [&](int64_t v) { return m_.mdValue(m_.constInt(i32_, v)); };
  bool vertex = model_ != spv::ExecutionModelFragment;  // vertex or post-tessellation vertex

  // Classify the module-scope variables.
  std::map<uint32_t, uint32_t> inputs, outputs;  // location -> variable
  uint32_t vertexIndex = kNone, primitiveId = kNone, fragCoord = kNone, frontFacing = kNone, perVertex = kNone,
           fragDepth = kNone, sampleMask = kNone, sampleMaskIn = kNone, sampleId = kNone, zpdCounter = kNone,
           linearBarycentric = kNone;
  for (uint32_t var : globals_) {
    const SType& pt = type(varType_[var]);
    const Decorations& d = decor_[var];
    std::string what = "variable %" + std::to_string(var);
    switch (pt.storage) {
      case spv::StorageClassInput:
        if (vertex && !tese() && d.builtin == spv::BuiltInVertexIndex) vertexIndex = var;
        else if (tese() && d.builtin == spv::BuiltInTessCoord) tessCoord_ = var;
        else if (tese() && d.builtin == spv::BuiltInPrimitiveId) primitiveId = var;
        else if (tese() && d.builtin == kNone && d.location != kNone && controlPointInput_ == kNone)
          controlPointInput_ = var;
        // A Sample-decorated gl_FragCoord is a plain position: Metal's [[position]] has no sample variant.
        else if (!vertex && d.builtin == spv::BuiltInFragCoord) fragCoord = var;
        else if (!vertex && d.builtin == spv::BuiltInSampleMask)
          sampleMaskIn = var;
        else if (!vertex && d.builtin == spv::BuiltInSampleId)
          sampleId = var;
        else if (!vertex && d.builtin == spv::BuiltInFrontFacing) frontFacing = var;
        else if (!vertex && d.builtin == spv::BuiltInBaryCoordKHR)
          fail("perspective barycentric input is not supported");
        else if (!vertex && d.builtin == spv::BuiltInBaryCoordNoPerspKHR)
          linearBarycentric = var;
        else if (!vertex && d.builtin == kNone && d.location != kNone) inputs[d.location] = var;
        else fail("input " + what);
        break;
      case spv::StorageClassOutput:
        if (vertex && type(pt.elem).op == spv::OpTypeStruct) perVertex = var;
        else if (!vertex && d.builtin == spv::BuiltInFragDepth) fragDepth = var;
        else if (!vertex && d.builtin == spv::BuiltInSampleMask) sampleMask = var;
        else if (d.builtin == kNone && d.location != kNone) outputs[d.location] = var;
        else fail("output " + what);
        break;
      case spv::StorageClassUniform:
        if (isFunctionConstant(var)) {
          const SType& block = type(pt.elem);
          if (block.op != spv::OpTypeStruct || block.members.size() != 1 ||
              type(block.members[0]).op != spv::OpTypeVector || type(block.members[0]).llvm != m_.vecTy(i32_, 4))
            fail("function constant block of " + what + " is not a struct of one uint4");
          if (d.binding > 0xFFFF) fail("function constant index of " + what);
          break;
        }
        if (d.set != options_.graphics.constantsSet || d.binding >= options_.graphics.constantBufferCount) fail("uniform " + what);
        break;
      case spv::StorageClassStorageBuffer:
        if (!vertex && options_.graphics.occlusionCounterBuffer >= 0 &&
            d.set == options_.graphics.occlusionCounterSet && d.binding == options_.graphics.occlusionCounterBinding &&
            zpdCounter == kNone)
          zpdCounter = var;
        else if (storageSlot(d) >= 0) {
        } else if (d.set != options_.graphics.sharedMemorySet || d.binding != options_.graphics.sharedMemoryBinding)
          fail("storage buffer " + what);
        break;
      case spv::StorageClassUniformConstant: {
        if (d.set != (vertex ? options_.graphics.vertexTexturesSet : options_.graphics.fragmentTexturesSet)) fail("resource set of " + what);
        uint16_t op = type(pt.elem).op;
        if (op == spv::OpTypeSampler) usesSamplers_ = true;
        else if (op == spv::OpTypeImage) usesTextures_ = true;
        else fail("resource " + what);
        break;
      }
      default: fail("storage class " + std::to_string(pt.storage) + " of " + what);
    }
  }

  // Arguments: stage inputs, then the argument buffer and heaps.
  if (vertexIndex != kNone) {
    argVertexId_ = int(addArg(i32_, {S("air.vertex_id"), S("air.arg_type_name"), S("uint"), S("air.arg_name"),
                                     S("vertex_id")}));
    if (options_.vertexIndexRelativeToBase) argBaseVertex_ = int(addArg(i32_, {S("air.base_vertex"), S("air.arg_type_name"), S("uint"),
                                       S("air.arg_name"), S("base_vertex")}));
    argOfInput_[vertexIndex] = -1;  // vertex_id - base_vertex, see buildEntry
  }
  if (tese()) {
    bool quad = patchDomain_ == PatchDomain::Quad;
    argPatchId_ =
        int(addArg(i32_, {S("air.patch_id"), S("air.arg_type_name"), S("uint"), S("air.arg_name"), S("patch_id")}));
    int position = int(addArg(m_.vecTy(f32_, quad ? 2 : 3),
                              {S("air.position_in_patch"), S("air.arg_type_name"), S(quad ? "float2" : "float3"),
                               S("air.arg_name"), S("position_in_patch")}));
    if (tessCoord_ != kNone) argOfInput_[tessCoord_] = position;
    if (primitiveId != kNone) argOfInput_[primitiveId] = argPatchId_;
    if (controlPointInput_ != kNone) {
      if (options_.graphics.controlPointBuffer < 0) fail("control point input without GraphicsBindings::controlPointBuffer");
      Type* t = llvmType(type(varType_[controlPointInput_]).elem);
      if (t->kind != Type::Array || t->elem != f32_) fail("control point input type");
      controlPoints_ = t->n;
      argControlPoints_ = int(addArg(
          m_.ptrTy(f32_, 1), {S("air.buffer"), S("air.location_index"), I(options_.graphics.controlPointBuffer), I(1), S("air.read"),
                              S("air.address_space"), I(1), S("air.arg_type_size"), I(4), S("air.arg_type_align_size"),
                              I(4), S("air.arg_type_name"), S("float"), S("air.arg_name"), S("control_points")}));
    }
  }
  for (auto& [location, var] : inputs) {
    const Decorations& d = decor_[var];
    Type* t = llvmType(type(varType_[var]).elem);
    if (!isFloatVector(t) && !isIntVector(t)) fail("interpolator type");
    // Vulkan requires integer inputs to be Flat.
    if (isIntVector(t) && !d.flat) fail("integer fragment input without Flat at location " + std::to_string(location));
    std::string name = "texcoord" + std::to_string(location);
    // Flat is the single air.flat qualifier in place of the sampling and perspective pair.
    std::vector<Metadata*> md = {S("air.fragment_input"), S("user(" + name + ")")};
    if (d.flat) {
      md.push_back(S("air.flat"));
    } else {
      md.push_back(S(d.sample ? "air.sample" : d.centroid ? "air.centroid" : "air.center"));
      md.push_back(S(d.noperspective ? "air.no_perspective" : "air.perspective"));
    }
    md.insert(md.end(),
              {S("air.arg_type_name"), S(varyingTypeName(type(varType_[var]).elem)), S("air.arg_name"), S(name)});
    argOfInput_[var] = int(addArg(t, md));
  }
  if (fragCoord != kNone)
    argOfInput_[fragCoord] = int(addArg(v4f32_, {S("air.position"), S("air.center"), S("air.no_perspective"),
                                                 S("air.arg_type_name"), S("float4"), S("air.arg_name"),
                                                 S("position")}));
  if (frontFacing != kNone)
    argOfInput_[frontFacing] = int(addArg(i1_, {S("air.front_facing"), S("air.arg_type_name"), S("bool"),
                                                S("air.arg_name"), S("front_facing")}));
  if (linearBarycentric != kNone) {
    const Decorations& d = decor_[linearBarycentric];
    if (llvmType(type(varType_[linearBarycentric]).elem) != m_.vecTy(f32_, 3))
      fail("barycentric input must be float3");
    if (d.centroid || d.sample || d.flat) fail("barycentric input requires center sampling");
    argOfInput_[linearBarycentric] = int(addArg(m_.vecTy(f32_, 3),
        {S("air.barycentric_coord"), S("air.center"), S("air.no_perspective"),
         S("air.arg_type_name"), S("float3"), S("air.arg_name"), S("linear_barycentrics")}));
  }
  addSampleInputs(sampleId, sampleMaskIn);
  auto indirect = [&](int location, const char* typeName) -> std::vector<Metadata*> {
    return {S("air.indirect_buffer"), S("air.location_index"), I(location), I(1), S("air.read"),
            S("air.address_space"), I(2), S("air.struct_type_info"), m_.mdNode({}), S("air.arg_type_name"),
            S(typeName), S("air.arg_name"), S(typeName)};
  };
  argTlab_ = int(addArg(m_.ptrTy(tlabTy_, 2), indirect(options_.graphics.argumentBuffer, "struct.ArgumentBuffer")));
  if (usesTextures_)
    argResHeap_ =
        int(addArg(m_.ptrTy(resHeapTy_, 2), indirect(options_.graphics.resourceHeapBuffer, "struct.ResourceHeap")));
  if (usesSamplers_)
    argSmpHeap_ =
        int(addArg(m_.ptrTy(smpHeapTy_, 2), indirect(options_.graphics.samplerHeapBuffer, "struct.SamplerHeap")));
  if (zpdCounter != kNone)
    argOfInput_[zpdCounter] = int(addArg(
        m_.ptrTy(i8_, 1), {S("air.buffer"), S("air.location_index"), I(options_.graphics.occlusionCounterBuffer), I(1), S("air.read_write"),
                           S("air.address_space"), I(1), S("air.arg_type_size"), I(1), S("air.arg_type_align_size"),
                           I(1), S("air.arg_type_name"), S("uchar"), S("air.arg_name"), S("zpd_counter")}));

  // Outputs: varyings or colour targets in Location order, then position / depth / coverage. Unwritten
  // builtins are dropped when the module does not write them.
  std::vector<Type*> retTypes;
  auto addOutput = [&](uint32_t var, int member, Type* t, std::vector<Metadata*> md, bool position, int clipFirst = -1,
                       int cullMember = -1) {
    outputVars_.push_back(var);
    outputs_.push_back({nullptr, t, std::move(md), position, clipFirst, cullMember});
    outputMember_.push_back(member);
    retTypes.push_back(t);
  };
  for (auto& [location, var] : outputs) {
    Type* t = llvmType(type(varType_[var]).elem);
    std::string n = std::to_string(location);
    if (vertex) {
      if (!isFloatVector(t) && !isIntVector(t)) fail("output type");
      addOutput(var, -1, t,
                {S("air.vertex_output"), S("user(texcoord" + n + ")"), S("air.arg_type_name"),
                 S(varyingTypeName(type(varType_[var]).elem)), S("air.arg_name"), S("texcoord" + n)},
                false);
    } else {
      if (!isFloatVector(t) && !isIntVector(t)) fail("output type");
      addOutput(var, -1, t,
                {S("air.render_target"), I(location), I(0), S("air.arg_type_name"),
                 S(varyingTypeName(type(varType_[var]).elem)), S("air.arg_name"), S("rt" + n)},
                false);
    }
  }
  if (perVertex != kNone) {
    uint32_t block = type(varType_[perVertex]).elem;
    const MemberDecorations& md = memberDecor_[block];
    int position = -1, clip = -1, cull = -1;
    for (size_t i = 0; i < md.builtin.size(); ++i) {
      if (md.builtin[i] == spv::BuiltInPosition) position = int(i);
      else if (md.builtin[i] == spv::BuiltInClipDistance) clip = int(i);
      else if (md.builtin[i] == spv::BuiltInCullDistance)
        cull = int(i);
      else if (md.builtin[i] != kNone) fail("vertex output builtin " + std::to_string(md.builtin[i]));
    }
    if (position < 0 || type(block).members.size() != 1u + (clip >= 0) + (cull >= 0)) fail("gl_PerVertex layout");
    std::vector<Metadata*> m = {S("air.position")};
    if (decor_[perVertex].invariant) m.push_back(S("air.invariant"));
    m.insert(m.end(), {S("air.arg_type_name"), S("float4"), S("air.arg_name"), S("position")});
    addOutput(perVertex, position, v4f32_, m, true);
    if (clip >= 0 || cull >= 0) {
      // Clip distances are one [N x float] air.clip_distance output (Apple's compiler writes this member for an MSL
      // `float x [[clip_distance]][N]`, N at most 8, with no companion varyings). Metal has no cull-distance
      // attribute; cull distances are the tail of the same array, see clipAndCullDistances.
      unsigned nClip = clip >= 0 ? llvmType(type(block).members[clip])->n : 0;
      unsigned nCull = cull >= 0 ? llvmType(type(block).members[cull])->n : 0;
      if (nClip + nCull > 8) fail("more than 8 clip and cull distances");
      addOutput(perVertex, clip, m_.arrayTy(f32_, nClip + nCull),
                {S("air.clip_distance"), S("air.clip_distance_array_size"), I(nClip + nCull), S("air.arg_type_name"),
                 S("float"), S("air.arg_name"), S("clip_planes")},
                false, -1, cull);
    }
  }
  if (fragDepth != kNone && stored_[fragDepth])
    addOutput(fragDepth, -1, f32_,
              {S("air.depth"), S("air.depth_qualifier"), S(depthQualifier_), S("air.arg_type_name"), S("float"),
               S("air.arg_name"), S("depth")},
              false);
  if (sampleMask != kNone && stored_[sampleMask])
    addOutput(sampleMask, 0, i32_, {S("air.sample_mask"), S("air.arg_type_name"), S("uint"), S("air.arg_name"),
                                    S("sample_mask")}, false);

  if (!retTypes.empty()) {
    retTy_ = m_.structTy("ret_main_ty");
    m_.setBody(retTy_, retTypes);
  }
  main_ = m_.addFunction("main", m_.fnTy(retTy_ ? retTy_ : m_.voidTy(), argTypes_));
  main_->unnamedAddr = 2;
}

// ---------------------------------------------------------------- sample rate

// The pixel shader's sample inputs: SampleId is MSL [[sample_id]], SampleMaskIn is [[sample_mask]]. Taking the sample
// index alone makes a fragment function run once per sample, which is what SampleId requires (Vulkan: sample shading);
// [[position]] stays at the pixel centre. tests/frag_gpu.mm checks the per-sample results on a 4x MSAA target.
// Metal places no order constraint on these arguments.
void Translator::addSampleInputs(uint32_t sampleId, uint32_t sampleMaskIn) {
  auto S = [&](const std::string& s) { return m_.mdString(s); };
  if (sampleId != kNone) {
    if (llvmType(type(varType_[sampleId]).elem) != i32_) fail("gl_SampleID type");
    argOfInput_[sampleId] =
        int(addArg(i32_, {S("air.sample_id"), S("air.arg_type_name"), S("uint"), S("air.arg_name"), S("sample_id")}));
  }
  if (sampleMaskIn != kNone) {
    Type* t = llvmType(type(varType_[sampleMaskIn]).elem);
    if (t->kind != Type::Array || t->n != 1 || t->elem != i32_) fail("gl_SampleMaskIn type");
    argOfInput_[sampleMaskIn] = int(
        addArg(i32_, {S("air.sample_mask_in"), S("air.arg_type_name"), S("uint"), S("air.arg_name"), S("mask_in")}));
  }
}

// ---------------------------------------------------------------- compute

const ResourceBinding& Translator::binding(uint32_t var) const {
  const Decorations& d = decor_[var];
  for (const ResourceBinding& b : options_.bindings)
    if (b.set == d.set && b.binding == d.binding) return b;
  fail("no binding for descriptor set " + std::to_string(d.set) + " binding " + std::to_string(d.binding));
}

// The arguments of the kernel: one per resource and builtin, with the metadata MSL gives a plain
// `device uchar*` / `constant uchar*` / `texture2d<T, access>` / `uint3 [[builtin]]` argument.
void Translator::buildComputeSignature() {
  auto S = [&](const std::string& s) { return m_.mdString(s); };
  auto I = [&](int64_t v) { return m_.mdValue(m_.constInt(i32_, v)); };
  for (uint32_t var : globals_) {
    const SType& pt = type(varType_[var]);
    const Decorations& d = decor_[var];
    std::string what = "variable %" + std::to_string(var);
    auto bufferArg = [&](unsigned index, unsigned addrSpace, bool readOnly) {
      argOfInput_[var] = int(
          addArg(m_.ptrTy(i8_, addrSpace),
                 {S("air.buffer"), S("air.location_index"), I(index), I(1), S(readOnly ? "air.read" : "air.read_write"),
                  S("air.address_space"), I(addrSpace), S("air.arg_type_size"), I(1), S("air.arg_type_align_size"),
                  I(1), S("air.arg_type_name"), S("uchar"), S("air.arg_name"), S("buffer" + std::to_string(index))}));
    };
    switch (pt.storage) {
      case spv::StorageClassInput: {
        const char* air = nullptr;
        bool scalar = false;
        switch (d.builtin) {
          case spv::BuiltInGlobalInvocationId:
            air = "air.thread_position_in_grid";
            break;
          case spv::BuiltInLocalInvocationId:
            air = "air.thread_position_in_threadgroup";
            break;
          case spv::BuiltInWorkgroupId:
            air = "air.threadgroup_position_in_grid";
            break;
          case spv::BuiltInNumWorkgroups:
            air = "air.threadgroups_per_grid";
            break;
          case spv::BuiltInLocalInvocationIndex:
            air = "air.thread_index_in_threadgroup";
            scalar = true;
            break;
          default:
            break;
        }
        if (!air) fail("input " + what);
        Type* t = llvmType(pt.elem);
        if (t != (scalar ? i32_ : m_.vecTy(i32_, 3))) fail("type of builtin " + what);
        argOfInput_[var] = int(addArg(t, {S(air), S("air.arg_type_name"), S(scalar ? "uint" : "uint3"),
                                          S("air.arg_name"), S(std::string(air).substr(4))}));
        break;
      }
      case spv::StorageClassStorageBuffer:
        bufferArg(binding(var).buffer, 1, memberDecor_[pt.elem].nonWritable || d.nonWritable);
        break;
      case spv::StorageClassUniform:
        // BufferBlock is the SPIR-V 1.0 spelling of a StorageBuffer block; a plain Block is a uniform buffer.
        if (decor_[pt.elem].bufferBlock)
          bufferArg(binding(var).buffer, 1, memberDecor_[pt.elem].nonWritable || d.nonWritable);
        else
          bufferArg(binding(var).buffer, 2, true);
        break;
      case spv::StorageClassPushConstant:
        if (options_.pushConstantBuffer < 0) fail("push constants without Options::pushConstantBuffer");
        bufferArg(uint32_t(options_.pushConstantBuffer), 2, true);
        break;
      case spv::StorageClassUniformConstant: {
        const SType& image = type(pt.elem);
        if (image.op != spv::OpTypeImage) fail("resource " + what);
        const SType& texel = type(image.elem);
        std::string component = texel.op == spv::OpTypeFloat ? "float" : texel.isSigned ? "int" : "uint";
        // A multisampled texture can only be declared read; a plain one is sampled-image access.
        const char* access = image.multisampled ? "read" : "sample";
        unsigned index = binding(var).texture;
        Type*& texTy = image.multisampled ? tex2dMs_ : tex2d_;
        if (!texTy) texTy = m_.structTy(image.multisampled ? "struct._texture_2d_ms_t" : "struct._texture_2d_t");
        argOfInput_[var] = int(addArg(
            m_.ptrTy(texTy, 1),
            {S("air.texture"), S("air.location_index"), I(index), I(1), S(std::string("air.") + access),
             S("air.arg_type_name"),
             S(std::string(image.multisampled ? "texture2d_ms<" : "texture2d<") + component + ", " + access + ">"),
             S("air.arg_name"), S("texture" + std::to_string(index))}));
        break;
      }
      case spv::StorageClassWorkgroup:
      case spv::StorageClassPrivate:
        break;
      default:
        fail("storage class " + std::to_string(pt.storage) + " of " + what);
    }
  }
  main_ = m_.addFunction("main", m_.fnTy(m_.voidTy(), argTypes_));
  main_->attrs = m_.attrList({Attr::get("nounwind"), Attr::str("frame-pointer", "all"), Attr::str("no-builtins")});
}

void Translator::buildComputeEntry() {
  entry_ = m_.addBlock(main_, "entry");
  exit_ = m_.addBlock(main_, "exit");
  entryB_ = std::make_unique<Builder>(m_, entry_);
  b_ = std::make_unique<Builder>(m_, entry_);
  for (uint32_t var : globals_) {
    const SType& pt = type(varType_[var]);
    Pointer& p = ptrs_[var];
    p.type = pt.elem;
    p.variable = var;
    switch (pt.storage) {
      case spv::StorageClassStorageBuffer:
      case spv::StorageClassUniform:
      case spv::StorageClassPushConstant:
        p.kind = Pointer::Buffer;
        p.ptr = main_->args[argOfInput_.at(var)];
        p.addrSpace = p.ptr->type->n;
        p.offset = m_.constInt(i64_, 0);
        break;
      case spv::StorageClassUniformConstant:
        p.kind = Pointer::Resource;
        p.ptr = main_->args[argOfInput_.at(var)];
        break;
      case spv::StorageClassWorkgroup: {
        // Threadgroup memory: an internal module global, no argument.
        Type* t = llvmType(pt.elem);
        p.kind = Pointer::Private;
        p.addrSpace = 3;
        p.ptr = m_.addGlobal("wg" + std::to_string(var), t, 3, m_.constUndef(t), 16);
        break;
      }
      case spv::StorageClassPrivate:
      case spv::StorageClassInput: {
        Type* t = llvmType(pt.elem);
        p.kind = Pointer::Private;
        p.ptr = entryB_->allocaInst(t, 16);
        Value* init = pt.storage == spv::StorageClassInput ? inputValue(var, t)
                      : varInit_.count(var)                ? val(varInit_.at(var))
                                                           : m_.constNull(t);
        entryB_->store(init, p.ptr, 4);
        break;
      }
    }
  }
}

// ---------------------------------------------------------------- entry and exit

Value* Translator::tlabSlot(uint32_t offset, Type* pointee) {
  Value* bytes = entryB_->cast(BitCast, main_->args[argTlab_], m_.ptrTy(i8_, 2));
  Value* slot = offset ? entryB_->gep(i8_, bytes, {m_.constInt(i64_, offset)}, true) : bytes;
  slot = entryB_->cast(BitCast, slot, m_.ptrTy(pointee, 2));
  return entryB_->load(pointee, slot, 8);
}

void Translator::buildEntry() {
  if (compute()) return buildComputeEntry();
  bool vertex = model_ != spv::ExecutionModelFragment;  // vertex or post-tessellation vertex
  entry_ = m_.addBlock(main_, "entry");
  exit_ = m_.addBlock(main_, "exit");
  entryB_ = std::make_unique<Builder>(m_, entry_);
  b_ = std::make_unique<Builder>(m_, entry_);
  for (uint32_t var : globals_) {
    const SType& pt = type(varType_[var]);
    Pointer& p = ptrs_[var];
    p.type = pt.elem;
    p.variable = var;
    switch (pt.storage) {
      case spv::StorageClassUniform:
        if (isFunctionConstant(var)) {
          p.kind = Pointer::FunctionConstant;
          break;
        }
        p.kind = Pointer::Buffer;
        p.addrSpace = 2;
        p.ptr = tlabSlot(options_.graphics.constantBaseOffset + options_.graphics.constantStride * decor_[var].binding, m_.ptrTy(i8_, 2));
        p.offset = m_.constInt(i64_, 0);
        break;
      case spv::StorageClassStorageBuffer: {
        bool readOnly = memberDecor_[pt.elem].nonWritable;
        p.kind = Pointer::Buffer;
        p.addrSpace = 1;
        auto zpd = argOfInput_.find(var);
        int slot = storageSlot(decor_[var]);
        p.ptr = slot >= 0 ? tlabSlot(options_.graphics.storageBuffers[slot].argumentBufferOffset, m_.ptrTy(i8_, 1))
                : zpd != argOfInput_.end() ? main_->args[zpd->second]
                                           : tlabSlot(readOnly ? options_.graphics.sharedMemoryReadOffset
                                                               : options_.graphics.sharedMemoryWriteOffset,
                                                      m_.ptrTy(i8_, 1));
        p.offset = m_.constInt(i64_, 0);
        break;
      }
      case spv::StorageClassUniformConstant: p.kind = Pointer::Resource; break;
      case spv::StorageClassInput:
      case spv::StorageClassOutput: {
        Type* t = llvmType(pt.elem);
        p.kind = Pointer::Private;
        p.ptr = entryB_->allocaInst(t, 16);
        if (pt.storage == spv::StorageClassOutput) {
          entryB_->store(m_.constNull(t), p.ptr, 16);
        } else {
          entryB_->store(inputValue(var, t), p.ptr, 4);
        }
        break;
      }
    }
  }
  if (usesTextures_ || usesSamplers_)
    textureIndices_ = tlabSlot(vertex ? options_.graphics.vertexTextureIndicesOffset : options_.graphics.fragmentTextureIndicesOffset, m_.ptrTy(i8_, 1));
}

// The value an input variable starts with.
Value* Translator::inputValue(uint32_t var, Type* t) {
  Builder& b = *entryB_;
  if (var == controlPointInput_) {
    // Control point k of patch p is element p * N + k of the control-point buffer.
    Value* first = b.binop(Mul, main_->args[argPatchId_], cint(int(controlPoints_)));
    Value* v = m_.constUndef(t);
    for (unsigned k = 0; k < controlPoints_; ++k) {
      Value* index = b.cast(ZExt, k ? b.binop(Add, first, cint(int(k))) : first, i64_);
      v = b.insertValue(v, b.load(f32_, b.gep(f32_, main_->args[argControlPoints_], {index}, true), 4), {k});
    }
    return v;
  }
  int arg = argOfInput_.at(var);
  if (arg < 0) {
    if (options_.vertexIndexRelativeToBase) return b.binop(Sub, main_->args[argVertexId_], main_->args[argBaseVertex_]);
    return main_->args[argVertexId_];
  }
  Value* v = main_->args[arg];
  if (t->kind == Type::Array) return b.insertValue(m_.constUndef(t), v, {0});  // gl_SampleMaskIn[1]
  if (var == tessCoord_ && patchDomain_ == PatchDomain::Quad) {
    // Vulkan's TessCoord for quads is (u, v, 0).
    Constant* mask =
        m_.constAggregate(m_.vecTy(i32_, 3), {m_.constInt(i32_, 0), m_.constInt(i32_, 1), m_.constInt(i32_, 2)});
    v = b.shuffleVector(v, m_.constNull(v->type), mask);
  }
  return v;
}

void Translator::buildExit() {
  b_->setInsertBlock(exit_);
  if (!retTy_) {
    b_->retVoid();
    return;
  }
  Value* agg = m_.constUndef(retTy_);
  for (size_t i = 0; i < outputs_.size(); ++i) {
    const Pointer& p = ptrs_[outputVars_[i]];
    Value* v;
    if (outputs_[i].cullMember >= 0) {
      v = clipAndCullDistances(p, outputMember_[i], outputs_[i].cullMember, outputs_[i].type);
      agg = b_->insertValue(agg, v, {uint32_t(i)});
      continue;
    }
    Value* ptr = p.ptr;
    if (outputMember_[i] >= 0)
      ptr = b_->gep(llvmType(p.type), ptr, {cint(0), cint(outputMember_[i])}, true);
    if (outputs_[i].clipFirst >= 0) {
      Type* array = llvmType(type(p.type).members[outputMember_[i]]);
      Value* distances = b_->load(array, ptr, 4);
      Type* t = outputs_[i].type;
      unsigned first = unsigned(outputs_[i].clipFirst);
      if (t->kind != Type::Vector) {
        v = b_->extractValue(distances, {first});
      } else {
        v = m_.constUndef(t);
        for (unsigned c = 0; c < t->n; ++c) v = b_->insertElement(v, b_->extractValue(distances, {first + c}), cint(c));
      }
    } else {
      v = b_->load(outputs_[i].type, ptr, 4);
    }
    if (outputs_[i].position && options_.positionInfToNan) {
      // positionInfToNan: an infinite x, y or z becomes NaN, so every non-finite coordinate culls the primitive (see
      // docs/bindings.md, "Non-finite positions"); w is left alone.
      for (unsigned c = 0; c < 3; ++c) {
        Value* bits = b_->cast(BitCast, b_->extractElement(v, cint(c)), i32_);
        Value* inf = b_->icmp(I_EQ, b_->binop(And, bits, cint(0x7F800000)), cint(0x7F800000));
        Value* fixed = b_->cast(BitCast, b_->select(inf, cint(0x7FC00000), bits), f32_);
        v = b_->insertElement(v, fixed, cint(c));
      }
    }
    agg = b_->insertValue(agg, v, {uint32_t(i)});
  }
  b_->ret(agg);
}

// Metal has clip distances only. A cull distance culls a primitive whose vertices are all negative and leaves
// every other primitive whole, which a clip distance cannot express directly (it would cut straddling primitives
// at the zero crossing), and a vertex function sees one vertex. What it can do is move the zero crossing next to
// the vertices that are negative: the distance becomes -1 where the cull distance is negative and 2^30 elsewhere.
// The interpolated value is negative only within about 2^-30 (relative to the primitive) of the negative vertices:
// a primitive with all of them negative is clipped completely, any other keeps all its area but a sliver far below
// the rasteriser's own precision. The distances follow the real clip distances in air.clip_distance.
Value* Translator::clipAndCullDistances(const Pointer& p, int clipMember, int cullMember, Type* result) {
  Value* r = m_.constUndef(result);
  unsigned k = 0;
  auto member = [&](int m) {
    Value* ptr = b_->gep(llvmType(p.type), p.ptr, {cint(0), cint(m)}, true);
    return b_->load(llvmType(type(p.type).members[m]), ptr, 4);
  };
  if (clipMember >= 0) {
    Value* clip = member(clipMember);
    for (unsigned e = 0; e < clip->type->n; ++e) r = b_->insertValue(r, b_->extractValue(clip, {e}), {k++});
  }
  Value* cull = member(cullMember);
  Value* negative = m_.constFPBits(f32_, 0xBF800000);  // -1
  Value* positive = m_.constFPBits(f32_, 0x4E800000);  // 2^30
  for (unsigned e = 0; e < cull->type->n; ++e) {
    Value* d = b_->extractValue(cull, {e});
    r = b_->insertValue(r, b_->select(b_->fcmp(F_OLT, d, m_.constFPBits(f32_, 0)), negative, positive), {k++});
  }
  return r;
}

void Translator::finalizeMetadata() {
  auto S = [&](const std::string& s) { return m_.mdString(s); };
  auto I = [&](int64_t v) { return m_.mdValue(m_.constInt(i32_, v)); };
  auto N = [&](std::vector<Metadata*> ops) { return m_.mdNode(std::move(ops)); };
  auto flag = [&](int b, const char* k, int v) { return N({I(b), S(k), I(v)}); };
  m_.addNamedMD("llvm.module.flags",
                {flag(1, "wchar_size", 4), flag(7, "air.max_device_buffers", 31),
                 flag(7, "air.max_constant_buffers", 31), flag(7, "air.max_threadgroup_buffers", 31),
                 flag(7, "air.max_textures", 128), flag(7, "air.max_read_write_textures", 8),
                 flag(7, "air.max_samplers", 16)});
  std::vector<Metadata*> args, outs;
  for (size_t i = 0; i < argMD_.size(); ++i) {
    std::vector<Metadata*> ops = {I(int64_t(i))};
    ops.insert(ops.end(), argMD_[i].begin(), argMD_[i].end());
    args.push_back(N(ops));
  }
  if (compute()) {
    // As MSL with fast math off: denormal flushing and explicit workgroup size as the threadgroup
    // limit (max_total_threads_per_threadgroup) so the compiler can budget registers for it.
    m_.addNamedMD("air.kernel",
                  {N({m_.mdValue(main_), N({}), N(args),
                      N({S("air.max_work_group_size"), I(int64_t(localSize_[0]) * localSize_[1] * localSize_[2])})})});
    m_.addNamedMD("air.compile_options",
                  {N({S("air.compile.denorms_disable")}), N({S("air.compile.fast_math_disable")}),
                   N({S("air.compile.framebuffer_fetch_enable")})});
    m_.addNamedMD("llvm.ident", {N({S("spirv2air")})});
    m_.addNamedMD("air.version", {N({I(airVersion_.major), I(airVersion_.minor), I(0)})});
    m_.addNamedMD("air.language_version", {N({S("Metal"), I(3), I(1), I(0)})});
    return;
  }
  for (const Output& o : outputs_) outs.push_back(N(o.md));
  bool vertex = model_ != spv::ExecutionModelFragment;  // vertex or post-tessellation vertex
  std::vector<Metadata*> fn = {m_.mdValue(main_), N(outs), N(args)};
  if (!vertex && earlyFragmentTests_) fn.push_back(S("early_fragment_tests"));
  if (tese())
    fn.push_back(N({S("air.patch"), S(patchDomain_ == PatchDomain::Quad ? "quad" : "triangle"),
                    S("air.patch_control_point"), I(controlPoints_)}));
  m_.addNamedMD(vertex ? "air.vertex" : "air.fragment", {N(fn)});
  if (!fcGlobals_.empty()) {
    // Function constants the final code reads, by index; the entries Apple's compiler writes are {initializer, type
    // name, name, index, required}. An unread one is dropped, as Apple's compiler drops an unused constant.
    std::set<const Value*> read;
    for (const BasicBlock* b : main_->blocks)
      for (const Instruction* i : b->insts)
        if (i->op == Op::Load) read.insert(i->ops[0]);
    std::vector<Metadata*> entries;
    for (auto& [index, g] : fcGlobals_) {
      if (!read.count(g)) {
        m_.globals.erase(std::find(m_.globals.begin(), m_.globals.end(), g));
        continue;
      }
      std::string name = "spirv2air_fc_" + std::to_string(index);
      entries.push_back(N({m_.mdValue(g), S("uint4"), S(name), I(index), m_.mdValue(m_.constInt(i1_, 1))}));
      fcNames_.push_back(name);
    }
    for (size_t k = 0; k < fcNames_.size(); ++k)
      fcEntries_.push_back({fcNames_[k], 36 /*MTLDataTypeUInt4*/, uint16_t(std::stoul(fcNames_[k].substr(13))), 1});
    if (!entries.empty()) m_.addNamedMD("air.function_constants", entries);
  }
  m_.addNamedMD("llvm.ident", {N({S("spirv2air")})});
  m_.addNamedMD("air.version", {N({I(airVersion_.major), I(airVersion_.minor), I(0)})});
  m_.addNamedMD("air.language_version", {N({S("Metal"), I(3), I(1), I(0)})});
}

// ---------------------------------------------------------------- body

Value* Translator::val(uint32_t id) const {
  if (id >= vals_.size() || !vals_[id]) fail("value %" + std::to_string(id) + " used before definition");
  return vals_[id];
}

BasicBlock* Translator::block(uint32_t label) {
  if (label >= vals_.size()) fail("label %" + std::to_string(label) + " out of range");
  if (blocks_.empty()) blocks_.resize(vals_.size());
  BasicBlock*& bb = blocks_[label];
  if (!bb) bb = m_.addBlock(main_);
  return bb;
}

void Translator::translateFunction() {
  for (size_t i = functionBegin_ + 1; i < functionEnd_; ++i) {
    const Inst& in = insts_[i];
    switch (in.op) {
      case spv::OpLabel:
        if (firstLabel_ == kNone) firstLabel_ = in.operand(0);
        b_->setInsertBlock(block(in.operand(0)));
        break;
      case spv::OpVariable: {
        const SType& pt = type(in.operand(0));
        if (pt.storage != spv::StorageClassFunction) fail("function variable storage class");
        Pointer& p = ptrs_[in.operand(1)];
        p.kind = Pointer::Private;
        p.type = pt.elem;
        p.variable = in.operand(1);
        p.ptr = entryB_->allocaInst(llvmType(pt.elem), 16);
        if (in.wc > 4) entryB_->store(val(in.operand(3)), p.ptr, 4);
        break;
      }
      case spv::OpFunctionParameter: fail("function parameters");
      default: translate(in);
    }
  }
  entryB_->br(block(firstLabel_));
  for (auto& [phi, in] : phis_) {
    for (unsigned w = 3; w + 1 < in.wc; w += 2) {
      phi->ops.push_back(val(in.w[w]));
      phi->bbs.push_back(block(in.w[w + 1]));
    }
  }
}

// ---------------------------------------------------------------- memory

Value* Translator::addOffset(Value* offset, Value* add) {
  auto* a = static_cast<Constant*>(offset);
  auto* b = static_cast<Constant*>(add);
  if (offset->vk == Value::Const && add->vk == Value::Const) return m_.constInt(i64_, int64_t(a->bits + b->bits));
  return b_->binop(Add, offset, add);
}

Value* Translator::bufferAddress(const Pointer& p, Type* elem) {
  Value* addr = p.ptr;
  bool zero = p.offset->vk == Value::Const && static_cast<Constant*>(p.offset)->bits == 0;
  if (!zero) addr = b_->gep(i8_, addr, {p.offset}, false);
  return b_->cast(BitCast, addr, m_.ptrTy(elem, p.addrSpace));
}

Value* Translator::loadBuffer(uint32_t typeId, Value* base, unsigned addrSpace, Value* offset) {
  const SType& t = type(typeId);
  Pointer p;
  p.ptr = base;
  p.offset = offset;
  p.addrSpace = addrSpace;
  switch (t.op) {
    case spv::OpTypeBool: fail("bool in a buffer");
    case spv::OpTypeInt: case spv::OpTypeFloat: case spv::OpTypeVector:
      return b_->load(t.llvm, bufferAddress(p, t.llvm), 4);
    case spv::OpTypeArray: {
      Value* agg = m_.constUndef(llvmType(typeId));
      for (uint32_t i = 0; i < t.count; ++i) {
        Value* e = loadBuffer(t.elem, base, addrSpace,
                              addOffset(offset, m_.constInt(i64_, int64_t(i) * decor_[typeId].stride)));
        agg = b_->insertValue(agg, e, {i});
      }
      return agg;
    }
    case spv::OpTypeStruct: {
      Value* agg = m_.constUndef(llvmType(typeId));
      const MemberDecorations& md = memberDecor_[typeId];
      for (uint32_t i = 0; i < t.members.size(); ++i) {
        Value* e = loadBuffer(t.members[i], base, addrSpace, addOffset(offset, m_.constInt(i64_, md.offset.at(i))));
        agg = b_->insertValue(agg, e, {i});
      }
      return agg;
    }
    default: fail("buffer load of type op " + std::to_string(t.op));
  }
}

bool Translator::isFunctionConstant(uint32_t var) const {
  return options_.graphics.functionConstantSet >= 0 &&
         decor_[var].set == uint32_t(options_.graphics.functionConstantSet);
}

// The value of function constant `binding`. Apple's compiler reads a `constant uint4 x [[function_constant(N)]]`
// through an internal global that a static initializer fills from an "air.fc_initializer" constant, which Metal
// replaces with the value given at specialization. Here the shader reads that constant directly.
Value* Translator::functionConstant(uint32_t binding) {
  GlobalVariable*& g = fcGlobals_[binding];
  if (!g) {
    Type* t = m_.vecTy(i32_, 4);
    std::string name = "spirv2air_fc_" + std::to_string(binding);
    g = m_.addGlobal("_Z" + std::to_string(name.size()) + name + ".MTL_FC_INIT_" + std::to_string(binding) + "_Dv4_j",
                     t, 2, m_.constUndef(t), 16, Link_Internal, 1, true);
    g->externallyInitialized = true;
    g->section = "air.fc_initializer";
  }
  return b_->load(g->valueType, g, 16);
}

Value* Translator::load(const Pointer& p) {
  switch (p.kind) {
    case Pointer::Private: return b_->load(llvmType(p.type), p.ptr, 4);
    case Pointer::Buffer: return loadBuffer(p.type, p.ptr, p.addrSpace, p.offset);
    case Pointer::Resource: return loadResource(p);
    case Pointer::FunctionConstant: {
      Value* v = functionConstant(decor_[p.variable].binding);
      const SType& t = type(p.type);
      if (t.op == spv::OpTypeStruct) return b_->insertValue(m_.constUndef(llvmType(p.type)), v, {0});
      if (t.op == spv::OpTypeVector) return v;
      return b_->extractElement(v, p.fcIndex ? p.fcIndex : cint(p.fcComponent));
    }
    default: fail("load through an unknown pointer");
  }
}

void Translator::store(const Pointer& p, Value* v) {
  switch (p.kind) {
    case Pointer::Private: b_->store(v, p.ptr, 4); return;
    case Pointer::Buffer: {
      const SType& t = type(p.type);
      if (t.op != spv::OpTypeInt && t.op != spv::OpTypeFloat && t.op != spv::OpTypeVector) fail("aggregate store");
      b_->store(v, bufferAddress(p, t.llvm), 4);
      return;
    }
    default: fail("store through this pointer");
  }
}

Pointer Translator::accessChain(const Pointer& base, const uint32_t* idx, unsigned n) {
  Pointer r = base;
  uint32_t t = base.type;
  if (base.kind == Pointer::Private) {
    std::vector<Value*> gidx = {cint(0)};
    for (unsigned k = 0; k < n; ++k) {
      const SType& st = type(t);
      if (st.op == spv::OpTypeStruct) {
        uint32_t m = constU32(idx[k]);
        gidx.push_back(cint(m));
        t = st.members.at(m);
      } else if (st.op == spv::OpTypeArray || st.op == spv::OpTypeVector) {
        gidx.push_back(val(idx[k]));
        t = st.elem;
      } else {
        fail("access chain into type op " + std::to_string(st.op));
      }
    }
    r.ptr = b_->gep(llvmType(base.type), base.ptr, gidx, true);
  } else if (base.kind == Pointer::Buffer) {
    Value* off = base.offset;
    for (unsigned k = 0; k < n; ++k) {
      const SType& st = type(t);
      if (st.op == spv::OpTypeStruct) {
        uint32_t m = constU32(idx[k]);
        off = addOffset(off, m_.constInt(i64_, memberDecor_[t].offset.at(m)));
        t = st.members.at(m);
        continue;
      }
      uint32_t stride;
      if (st.op == spv::OpTypeArray || st.op == spv::OpTypeRuntimeArray) stride = decor_[t].stride;
      else if (st.op == spv::OpTypeVector) stride = 4;
      else fail("access chain into type op " + std::to_string(st.op));
      if (!stride) fail("array without ArrayStride");
      if (isConstant(idx[k])) {
        off = addOffset(off, m_.constInt(i64_, int64_t(constU32(idx[k])) * stride));
      } else {
        Value* i = b_->cast(ZExt, val(idx[k]), i64_);
        if (stride != 1) i = b_->binop(Mul, i, m_.constInt(i64_, stride));
        off = addOffset(off, i);
      }
      t = st.elem;
    }
    r.offset = off;
  } else if (base.kind == Pointer::FunctionConstant) {
    // Block member 0, then at most one component of the uint4.
    for (unsigned k = 0; k < n; ++k) {
      const SType& st = type(t);
      if (st.op == spv::OpTypeStruct) {
        if (!isConstant(idx[k]) || constU32(idx[k]) != 0) fail("function constant block member other than 0");
        t = st.members[0];
      } else if (st.op == spv::OpTypeVector) {
        if (isConstant(idx[k])) {
          if (constU32(idx[k]) >= st.count) fail("function constant component out of range");
          r.fcComponent = int(constU32(idx[k]));
        } else {
          r.fcIndex = b_->binop(And, val(idx[k]), cint(int(st.count) - 1));
        }
        t = st.elem;
      } else {
        fail("access chain into type op " + std::to_string(st.op) + " of a function constant");
      }
    }
  } else {
    fail("access chain into a resource");
  }
  r.type = t;
  return r;
}

Value* Translator::loadResource(const Pointer& p) {
  if (compute()) return p.ptr;  // the texture argument itself
  const Decorations& d = decor_[p.variable];
  const SType& t = type(p.type);
  bool sampler = t.op == spv::OpTypeSampler;
  // The stage's index buffer holds one {texture heap index, sampler heap index} pair per binding.
  Value* slot = b_->gep(i8_, textureIndices_, {m_.constInt(i64_, options_.graphics.textureIndexStride * d.binding +
                                  (sampler ? options_.graphics.samplerIndexOffset : options_.graphics.textureIndexOffset))}, true);
  Value* index = b_->load(i32_, b_->cast(BitCast, slot, m_.ptrTy(i32_, 1)), 4);
  Value* heap = b_->cast(BitCast, main_->args[sampler ? argSmpHeap_ : argResHeap_], m_.ptrTy(i8_, 2));
  Value* entry = b_->gep(i8_, heap, {b_->cast(SExt, b_->binop(Mul, index, cint(options_.graphics.descriptorStride)), i64_)}, true);
  uint32_t handleOffset = sampler ? options_.graphics.samplerHandleOffset : options_.graphics.textureHandleOffset;
  if (handleOffset) entry = b_->gep(i8_, entry, {m_.constInt(i64_, handleOffset)}, true);
  Type* handle =
      sampler ? m_.ptrTy(samplerTy_, 2) : m_.ptrTy(textureTypes_.at(t.dim | (t.arrayed ? kArrayedKey : 0)), 1);
  return b_->load(handle, b_->cast(BitCast, entry, m_.ptrTy(handle, 2)), 8);
}

// ---------------------------------------------------------------- arithmetic helpers

Value* Translator::splat(Value* s, unsigned n) {
  Type* vt = m_.vecTy(s->type, n);
  Value* v = b_->insertElement(m_.constUndef(vt), s, cint(0));
  std::vector<Constant*> zeros(n, m_.constInt(i32_, 0));
  return b_->shuffleVector(v, m_.constUndef(vt), m_.constAggregate(m_.vecTy(i32_, n), zeros));
}

Value* Translator::perComponent(std::vector<Value*> ops,
                                const std::function<Value*(const std::vector<Value*>&)>& f) {
  Type* t = ops[0]->type;
  if (t->kind != Type::Vector) return f(ops);
  Value* r = nullptr;
  std::vector<Value*> s;
  for (unsigned c = 0; c < t->n; ++c) {
    s.clear();
    for (Value* o : ops) s.push_back(o->type->kind == Type::Vector ? b_->extractElement(o, cint(c)) : o);
    Value* e = f(s);
    if (!r) r = m_.constUndef(m_.vecTy(e->type, t->n));
    r = b_->insertElement(r, e, cint(c));
  }
  return r;
}

Function* Translator::declare(const std::string& name, Type* ret, const std::vector<Type*>& params,
                              AttrClass cls) {
  Function*& f = decls_[name];
  if (f) return f;
  std::vector<Attr> a;
  // The attribute sets Apple's compiler writes on the declarations of these intrinsics for MSL (clean Q30): ALU, math,
  // bit and convert intrinsics are pure; derivatives, SIMD and barrier operations are convergent; implicit-LOD
  // samples and LOD queries are convergent reads of their arguments, while explicit-gradient samples, reads and
  // gathers are not convergent; atomics and discard only promise progress.
  switch (cls) {
    case AttrClass::Pure:
    case AttrClass::Bits:
      a = {Attr::get("mustprogress"), Attr::get("nofree"),   Attr::get("nosync"),
           Attr::get("nounwind"),     Attr::get("readnone"), Attr::get("willreturn")};
      break;
    case AttrClass::Texture:
      a = {Attr::get("argmemonly"), Attr::get("convergent"), Attr::get("mustprogress"), Attr::get("nofree"),
           Attr::get("nounwind"),   Attr::get("readonly"),   Attr::get("willreturn")};
      break;
    case AttrClass::TextureRead:
      a = {Attr::get("argmemonly"), Attr::get("mustprogress"), Attr::get("nofree"),
           Attr::get("nounwind"),   Attr::get("readonly"),     Attr::get("willreturn")};
      break;
    case AttrClass::Derivative:
    case AttrClass::Barrier:
      a = {Attr::get("convergent"), Attr::get("mustprogress"), Attr::get("nounwind"), Attr::get("willreturn")};
      break;
    case AttrClass::Discard:
    case AttrClass::Atomic:
      a = {Attr::get("mustprogress"), Attr::get("nounwind"), Attr::get("willreturn")};
      break;
  }
  f = m_.addFunction(name, m_.fnTy(ret, params));
  f->attrs = m_.attrList(a);
  f->unnamedAddr = 2;
  return f;
}

Value* Translator::intrinsic(const std::string& name, Type* ret, const std::vector<Value*>& args, AttrClass cls,
                             uint8_t fmf) {
  Function* f;
  if (auto it = decls_.find(name); it != decls_.end()) {
    f = it->second;
  } else {
    std::vector<Type*> params;
    for (Value* v : args) params.push_back(v->type);
    f = declare(name, ret, params, cls);
  }
  return b_->call(f, args, nullptr, "", fmf);
}

Value* Translator::fast(const char* op, std::vector<Value*> args) {
  // Compute shaders are compiled as MSL with fast math off (the dump shaders pack exact bit patterns): the precise
  // intrinsics, `air.<op>.f32`, with no fast-math flags.
  if (compute())
    return perComponent(std::move(args), [&](const std::vector<Value*>& s) {
      return intrinsic(std::string("air.") + op + ".f32", f32_, s, AttrClass::Pure);
    });
  return perComponent(std::move(args), [&](const std::vector<Value*>& s) {
    return intrinsic(std::string("air.fast_") + op + ".f32", f32_, s, AttrClass::Pure);
  });
}

// Nir shift semantics: the amount is taken modulo the bit width (LLVM would make an oversized shift poison).
Value* Translator::shiftAmount(Value* v) {
  if (v->vk == Value::Const && v->type == i32_) {
    auto* c = static_cast<Constant*>(v);
    if (c->ck == Constant::Int || c->ck == Constant::Null) return m_.constInt(i32_, int64_t(c->bits & 31));
  }
  Value* mask = cint(31);
  if (v->type->kind == Type::Vector) {
    std::vector<Constant*> ms(v->type->n, m_.constInt(i32_, 31));
    mask = m_.constAggregate(v->type, ms);
  }
  return b_->binop(And, v, mask);
}

// Guard the operands before division: a select after an invalid LLVM divide would still produce poison.
// SPIR-V leaves zero divisors and INT_MIN/-1 open. Use all-ones quotients for zero, the dividend for zero
// remainders, and INT_MIN and 0 as the quotient and remainder of INT_MIN/-1.
Value* Translator::integerDivide(Value* x, Value* y, uint16_t op) {
  return perComponent({x, y}, [&](const std::vector<Value*>& s) {
    bool sign = op == spv::OpSDiv || op == spv::OpSRem || op == spv::OpSMod;
    bool rem = op != spv::OpUDiv && op != spv::OpSDiv;
    Value* zero = b_->icmp(I_EQ, s[1], cint(0));
    Value* bad = zero;
    Value* overflow = nullptr;
    if (sign) {
      overflow = b_->binop(And, b_->icmp(I_EQ, s[0], cint(INT32_MIN)), b_->icmp(I_EQ, s[1], cint(-1)));
      bad = b_->binop(Or, bad, overflow);
    }
    Value* divisor = b_->select(bad, cint(1), s[1]);
    Value* r = b_->binop(rem ? (sign ? SRem : URem) : (sign ? SDiv : UDiv), s[0], divisor);
    if (sign && rem) r = b_->select(overflow, cint(0), r);
    if (op == spv::OpSMod) {
      Value* differentSign = b_->icmp(I_SLT, b_->binop(Xor, s[0], s[1]), cint(0));
      Value* adjust = b_->binop(And, differentSign, b_->icmp(I_NE, r, cint(0)));
      r = b_->select(adjust, b_->binop(Add, r, s[1]), r);
    }
    return b_->select(zero, rem ? s[0] : cint(-1), r);
  });
}

Value* Translator::extendedInteger(const Inst& in) {
  Value *x = val(in.operand(2)), *y = val(in.operand(3));
  Value* lo = perComponent({x, y}, [&](const std::vector<Value*>& s) -> Value* {
    return b_->binop(in.op == spv::OpIAddCarry ? Add : in.op == spv::OpISubBorrow ? Sub : Mul, s[0], s[1]);
  });
  Value* hi = perComponent({x, y}, [&](const std::vector<Value*>& s) -> Value* {
    if (in.op == spv::OpIAddCarry) return b_->cast(ZExt, b_->icmp(I_ULT, b_->binop(Add, s[0], s[1]), s[0]), i32_);
    if (in.op == spv::OpISubBorrow) return b_->cast(ZExt, b_->icmp(I_ULT, s[0], s[1]), i32_);
    const char* name = in.op == spv::OpSMulExtended ? "air.mul_hi.s.i32" : "air.mul_hi.u.i32";
    return intrinsic(name, i32_, s, AttrClass::Pure);
  });
  Value* r = b_->insertValue(m_.constUndef(llvmType(in.operand(0))), lo, {0});
  return b_->insertValue(r, hi, {1});
}

Value* Translator::scalarCast(CastOpc opc, Value* v, Type* to) { return b_->cast(opc, v, to); }

// ---------------------------------------------------------------- instructions

void Translator::translate(const Inst& in) {
  auto res = [&]() { return in.operand(1); };
  auto set = [&](Value* v) { vals_[in.operand(1)] = v; };
  auto op = [&](unsigned i) { return val(in.operand(i)); };
  auto rt = [&]() { return llvmType(in.operand(0)); };
  auto bin = [&](BinOpc opc) { set(b_->binop(opc, op(2), op(3))); };
  auto icmp = [&](uint8_t pred) { set(b_->icmp(pred, op(2), op(3))); };
  auto fcmp = [&](uint8_t pred) { set(b_->fcmp(pred, op(2), op(3))); };
  auto cast = [&](CastOpc opc) { set(b_->cast(opc, op(2), rt())); };
  auto allOnes = [&](Type* t, int64_t v) -> Value* {
    if (t->kind != Type::Vector) return m_.constInt(t, v);
    std::vector<Constant*> e(t->n, m_.constInt(t->elem, v));
    return m_.constAggregate(t, e);
  };
  auto reduce = [&](BinOpc opc) {
    Value* v = op(2);
    if (v->type->kind != Type::Vector) return set(v);
    Value* r = b_->extractElement(v, cint(0));
    for (unsigned c = 1; c < v->type->n; ++c) r = b_->binop(opc, r, b_->extractElement(v, cint(c)));
    set(r);
  };
  switch (in.op) {
    case spv::OpNop: case spv::OpLine: case spv::OpNoLine: case spv::OpSelectionMerge: case spv::OpLoopMerge:
      break;
    case spv::OpUndef: set(m_.constUndef(rt())); break;

    // ---- memory
    case spv::OpLoad:
      set(load(ptrs_[in.operand(2)]));
      if (ptrs_[in.operand(2)].kind == Pointer::Resource) loadedImageType_[res()] = ptrs_[in.operand(2)].type;
      break;
    case spv::OpStore: store(ptrs_[in.operand(0)], op(1)); break;
    case spv::OpAccessChain:
    case spv::OpInBoundsAccessChain:
      ptrs_[res()] = accessChain(ptrs_[in.operand(2)], in.w + 4, in.wc - 4u);
      break;
    case spv::OpCopyLogical:
    case spv::OpCopyObject:
      if (ptrs_[in.operand(2)].kind != Pointer::Invalid) ptrs_[res()] = ptrs_[in.operand(2)];
      else set(op(2));
      break;

    // ---- composites
    case spv::OpCompositeExtract: {
      Value* v = op(2);
      for (unsigned w = 4; w < in.wc; ++w)
        v = v->type->kind == Type::Vector ? b_->extractElement(v, cint(in.w[w])) : b_->extractValue(v, {in.w[w]});
      set(v);
      break;
    }
    case spv::OpCompositeInsert: {
      Value* object = op(2);
      std::function<Value*(Value*, unsigned)> insert = [&](Value* agg, unsigned w) -> Value* {
        uint32_t i = in.w[w];
        bool vec = agg->type->kind == Type::Vector;
        if (w + 1 == in.wc) return vec ? b_->insertElement(agg, object, cint(i)) : b_->insertValue(agg, object, {i});
        Value* inner = vec ? b_->extractElement(agg, cint(i)) : b_->extractValue(agg, {i});
        inner = insert(inner, w + 1);
        return vec ? b_->insertElement(agg, inner, cint(i)) : b_->insertValue(agg, inner, {i});
      };
      set(insert(op(3), 5));
      break;
    }
    case spv::OpCompositeConstruct: {
      Type* t = rt();
      Value* r = m_.constUndef(t);
      if (t->kind == Type::Vector) {
        unsigned c = 0;
        for (unsigned w = 3; w < in.wc; ++w) {
          Value* part = val(in.w[w]);
          if (part->type->kind == Type::Vector) {
            for (unsigned k = 0; k < part->type->n; ++k)
              r = b_->insertElement(r, b_->extractElement(part, cint(k)), cint(c++));
          } else {
            r = b_->insertElement(r, part, cint(c++));
          }
        }
      } else {
        for (unsigned w = 3; w < in.wc; ++w) r = b_->insertValue(r, val(in.w[w]), {w - 3});
      }
      set(r);
      break;
    }
    case spv::OpVectorExtractDynamic:
    case spv::OpVectorInsertDynamic: {
      bool insert = in.op == spv::OpVectorInsertDynamic;
      Value *v = op(2), *index = op(insert ? 4 : 3);
      Value* valid = b_->icmp(I_ULT, index, cint(v->type->n));
      Value* safe = b_->select(valid, index, cint(0));
      // Out-of-range indices are undefined in SPIR-V; choose zero / unchanged, without LLVM poison.
      if (insert)
        set(b_->select(valid, b_->insertElement(v, op(3), safe), v));
      else
        set(b_->select(valid, b_->extractElement(v, safe), m_.constNull(rt())));
      break;
    }
    case spv::OpVectorShuffle: {
      Value *a = op(2), *b = op(3);
      unsigned na = a->type->n, n = in.wc - 5u;
      if (a->type == b->type) {
        std::vector<Constant*> mask;
        for (unsigned w = 5; w < in.wc; ++w)
          mask.push_back(in.w[w] == 0xFFFFFFFFu ? m_.constUndef(i32_) : m_.constInt(i32_, in.w[w]));
        set(b_->shuffleVector(a, b, m_.constAggregate(m_.vecTy(i32_, n), mask)));
      } else {
        Value* r = m_.constUndef(rt());
        for (unsigned w = 5; w < in.wc; ++w) {
          if (in.w[w] == 0xFFFFFFFFu) continue;
          Value* e = in.w[w] < na ? b_->extractElement(a, cint(in.w[w])) : b_->extractElement(b, cint(in.w[w] - na));
          r = b_->insertElement(r, e, cint(w - 5));
        }
        set(r);
      }
      break;
    }

    // ---- arithmetic
    case spv::OpIAdd: bin(Add); break;
    case spv::OpISub: bin(Sub); break;
    case spv::OpIMul: bin(Mul); break;
    case spv::OpUDiv:
    case spv::OpUMod:
    case spv::OpSDiv:
    case spv::OpSRem:
    case spv::OpSMod:
      set(integerDivide(op(2), op(3), in.op));
      break;
    case spv::OpIAddCarry:
    case spv::OpISubBorrow:
    case spv::OpUMulExtended:
    case spv::OpSMulExtended:
      set(extendedInteger(in));
      break;
    case spv::OpFRem:
      set(perComponent({op(2), op(3)}, [&](const std::vector<Value*>& s) {
        return intrinsic("air.fmod.f32", f32_, s, AttrClass::Pure);
      }));
      break;
    case spv::OpFMod: {
      Value* quotient = b_->binop(FDiv, op(2), op(3));
      Value* floor = perComponent({quotient}, [&](const std::vector<Value*>& s) {
        return intrinsic("air.floor.f32", f32_, s, AttrClass::Pure);
      });
      set(b_->binop(FSub, op(2), b_->binop(FMul, op(3), floor)));
      break;
    }
    case spv::OpFAdd:
    case spv::OpFSub:
    case spv::OpFMul: {
      Value *x = op(2), *y = op(3);
      {
        // MSL compiled with fast math off still contracts a*b+c inside one expression into fmuladd, which is what
        // SPIRV-Cross's output (single-use temporaries are inlined) gets. Reproduce it for single-use products, in
        // every stage: SPIR-V and D3D12 allow contraction unless the operation is NoContraction (precise).
        auto fused = [&](uint32_t mulId, Value* addend, bool negAddend, bool negProduct) -> Value* {
          auto it = mulOps_.find(mulId);
          if (it == mulOps_.end() || uses_[mulId] != 1) return nullptr;
          Value *ma = it->second.first, *mb = it->second.second;
          if (negProduct) ma = b_->fneg(ma);
          if (negAddend) addend = b_->fneg(addend);
          return perComponent({ma, mb, addend}, [&](const std::vector<Value*>& s) {
            return intrinsic("llvm.fmuladd.f32", f32_, s, AttrClass::Pure);
          });
        };
        if (in.op == spv::OpFMul) {
          // Explicit None grants no NSZ permission. The compatibility fma
          // shape below normalizes exact negative-zero products to +0.
          if (decor_[res()].noContraction && compute() && !decor_[res()].explicitMathMode) {
            // SPIRV-Cross spells a precise product fma(l, r, 0), which also turns -0 into +0; keep that. air.fma.f32 is
            // what Apple's compiler writes for an MSL fma() (it never emits llvm.fma; -fno-fast-math contracts a*b+c to
            // llvm.fmuladd instead).
            set(perComponent({x, y}, [&](const std::vector<Value*>& s) {
              return intrinsic("air.fma.f32", f32_, {s[0], s[1], m_.constFPBits(f32_, 0)}, AttrClass::Pure);
            }));
            break;
          }
          if (!decor_[res()].noContraction) mulOps_[res()] = {x, y};
        } else if (!decor_[res()].noContraction) {
          bool sub = in.op == spv::OpFSub;
          Value* r = fused(in.operand(2), y, sub, false);
          if (!r) r = fused(in.operand(3), x, false, sub);
          if (r) {
            set(r);
            break;
          }
        }
      }
      set(b_->binop(in.op == spv::OpFAdd ? FAdd : in.op == spv::OpFSub ? FSub : FMul, x, y));
      break;
    }
    case spv::OpFDiv: bin(FDiv); break;
    case spv::OpFNegate: set(b_->fneg(op(2))); break;
    case spv::OpSNegate: set(b_->binop(Sub, m_.constNull(rt()), op(2))); break;
    case spv::OpVectorTimesScalar: set(b_->binop(FMul, op(2), splat(op(3), rt()->n))); break;
    case spv::OpDot: {
      Value *a = op(2), *b = op(3);
      Value* r = nullptr;
      for (unsigned c = 0; c < a->type->n; ++c) {
        Value* p = b_->binop(FMul, b_->extractElement(a, cint(c)), b_->extractElement(b, cint(c)));
        r = r ? b_->binop(FAdd, r, p) : p;
      }
      set(r);
      break;
    }
    case spv::OpShiftLeftLogical: set(b_->binop(Shl, op(2), shiftAmount(op(3)))); break;
    case spv::OpShiftRightLogical: set(b_->binop(LShr, op(2), shiftAmount(op(3)))); break;
    case spv::OpShiftRightArithmetic: set(b_->binop(AShr, op(2), shiftAmount(op(3)))); break;
    case spv::OpBitwiseAnd: case spv::OpLogicalAnd: bin(And); break;
    case spv::OpBitwiseOr: case spv::OpLogicalOr: bin(Or); break;
    case spv::OpBitwiseXor: case spv::OpLogicalNotEqual: bin(Xor); break;
    case spv::OpNot: set(b_->binop(Xor, op(2), allOnes(rt(), -1))); break;
    case spv::OpLogicalNot: set(b_->binop(Xor, op(2), allOnes(rt(), 1))); break;
    case spv::OpLogicalEqual: icmp(I_EQ); break;
    case spv::OpBitReverse:
      set(perComponent({op(2)}, [&](const std::vector<Value*>& s) {
        return intrinsic("air.reverse_bits.i32", i32_, s, AttrClass::Pure);
      }));
      break;
    case spv::OpBitCount:
      set(perComponent({op(2)}, [&](const std::vector<Value*>& s) {
        return intrinsic("air.popcount.i32", i32_, s, AttrClass::Pure);
      }));
      break;
    case spv::OpBitFieldInsert:
      // Count 0 leaves Base unchanged; skip the guard when Count is a known non-zero constant.
      set(perComponent({op(2), op(3), op(4), op(5)}, [&](const std::vector<Value*>& s) {
        Value* r = intrinsic("air.insert_bits.u.i32", i32_, s, AttrClass::Bits);
        return countIsNonZeroConst(s[3]) ? r : b_->select(b_->icmp(I_EQ, s[3], cint(0)), s[0], r);
      }));
      break;
    case spv::OpBitFieldSExtract:
    case spv::OpBitFieldUExtract: {
      const char* name = in.op == spv::OpBitFieldSExtract ? "air.extract_bits.s.i32" : "air.extract_bits.u.i32";
      // Count 0 yields 0 (air.extract_bits leaves the result unspecified for it).
      set(perComponent({op(2), op(3), op(4)}, [&](const std::vector<Value*>& s) {
        Value* r = intrinsic(name, i32_, s, AttrClass::Bits);
        return countIsNonZeroConst(s[2]) ? r : b_->select(b_->icmp(I_EQ, s[2], cint(0)), cint(0), r);
      }));
      break;
    }

    // ---- conversions
    case spv::OpConvertFToU:
    case spv::OpConvertFToS:
      if (compute() && (op(2)->type == f32_ || (op(2)->type->kind == Type::Vector && op(2)->type->elem == f32_))) {
        // MSL's float -> int conversion saturates (and maps NaN to 0); fptoui/fptosi would be undefined out of range.
        bool sign = in.op == spv::OpConvertFToS;
        set(perComponent({op(2)}, [&](const std::vector<Value*>& s) {
          return intrinsic(sign ? "air.convert.s.i32.f.f32" : "air.convert.u.i32.f.f32", i32_, s, AttrClass::Pure);
        }));
      } else {
        cast(in.op == spv::OpConvertFToU ? FPToUI : FPToSI);
      }
      break;
    case spv::OpQuantizeToF16:
      set(perComponent({op(2)}, [&](const std::vector<Value*>& s) -> Value* {
        Value* rounded = b_->cast(FPExt, b_->cast(FPTrunc, s[0], f16_), f32_);
        Value* raw = b_->cast(BitCast, rounded, i32_);
        Value* magnitude = b_->binop(And, raw, cint(0x7FFFFFFF));
        Value* tiny = b_->icmp(I_ULT, magnitude, cint(0x38800000));
        return b_->cast(BitCast, b_->select(tiny, b_->binop(And, raw, cint(0x80000000)), raw), f32_);
      }));
      break;
    case spv::OpConvertSToF: cast(SIToFP); break;
    case spv::OpConvertUToF: cast(UIToFP); break;
    case spv::OpBitcast:
      if (op(2)->type == rt()) set(op(2));
      else cast(BitCast);
      break;

    // ---- comparisons and selection
    case spv::OpIEqual: icmp(I_EQ); break;
    case spv::OpINotEqual: icmp(I_NE); break;
    case spv::OpUGreaterThan: icmp(I_UGT); break;
    case spv::OpSGreaterThan: icmp(I_SGT); break;
    case spv::OpUGreaterThanEqual: icmp(I_UGE); break;
    case spv::OpSGreaterThanEqual: icmp(I_SGE); break;
    case spv::OpULessThan: icmp(I_ULT); break;
    case spv::OpSLessThan: icmp(I_SLT); break;
    case spv::OpULessThanEqual: icmp(I_ULE); break;
    case spv::OpSLessThanEqual: icmp(I_SLE); break;
    case spv::OpFOrdEqual: fcmp(F_OEQ); break;
    case spv::OpFOrdNotEqual: fcmp(F_ONE); break;
    case spv::OpFUnordNotEqual: fcmp(F_UNE); break;
    case spv::OpFOrdLessThan: fcmp(F_OLT); break;
    case spv::OpFOrdGreaterThan: fcmp(F_OGT); break;
    case spv::OpFOrdLessThanEqual: fcmp(F_OLE); break;
    case spv::OpFOrdGreaterThanEqual: fcmp(F_OGE); break;
    case spv::OpFUnordLessThan: fcmp(F_ULT); break;
    case spv::OpFUnordGreaterThan: fcmp(F_UGT); break;
    case spv::OpFUnordLessThanEqual: fcmp(F_ULE); break;
    case spv::OpFUnordGreaterThanEqual: fcmp(F_UGE); break;
    case spv::OpFUnordEqual: fcmp(F_UEQ); break;
    case spv::OpIsInf:
      set(perComponent({op(2)}, [&](const std::vector<Value*>& s) {
        Value* magnitude = b_->binop(And, b_->cast(BitCast, s[0], i32_), cint(0x7FFFFFFF));
        return b_->icmp(I_EQ, magnitude, cint(0x7F800000));
      }));
      break;
    case spv::OpIsNan: set(b_->fcmp(F_UNO, op(2), op(2))); break;
    case spv::OpAny: reduce(Or); break;
    case spv::OpAll: reduce(And); break;
    case spv::OpSelect: set(b_->select(op(2), op(3), op(4))); break;

    // ---- derivatives
    case spv::OpDPdx: case spv::OpDPdxCoarse: case spv::OpDPdxFine:
    case spv::OpDPdy: case spv::OpDPdyCoarse: case spv::OpDPdyFine: {
      bool x = in.op == spv::OpDPdx || in.op == spv::OpDPdxCoarse || in.op == spv::OpDPdxFine;
      set(perComponent({op(2)}, [&](const std::vector<Value*>& s) {
        return intrinsic(x ? "air.dfdx.f32" : "air.dfdy.f32", f32_, s, AttrClass::Derivative);
      }));
      break;
    }

    // ---- extended, images, atomics
    case spv::OpExtInst:
      if (in.operand(2) != glslExt_) fail("extended instruction set");
      set(extInst(in));
      break;
    case spv::OpSampledImage: {
      uint32_t image = type(in.operand(0)).elem;
      sampledImages_[res()] = {op(2), op(3)};
      sampledImageType_[res()] = image;
      break;
    }
    case spv::OpImageSampleImplicitLod:
    case spv::OpImageSampleExplicitLod: set(sample(in)); break;
    case spv::OpImage:
      if (!sampledImages_.count(in.operand(2))) fail("image without OpSampledImage");
      set(sampledImages_.at(in.operand(2)).first);
      loadedImageType_[res()] = sampledImageType_.at(in.operand(2));
      break;
    case spv::OpImageQuerySize:
    case spv::OpImageQuerySizeLod:
    case spv::OpImageQueryLevels:
    case spv::OpImageQuerySamples:
      set(imageQuery(in));
      break;
    case spv::OpImageQueryLod: {
      // (clamped, unclamped) level of detail, as two AIR queries.
      auto it = sampledImages_.find(in.operand(2));
      if (it == sampledImages_.end()) fail("query without OpSampledImage");
      const SType& image = type(sampledImageType_.at(in.operand(2)));
      // The query takes the coordinate without the array layer; Apple's compiler writes one pair per texture type.
      unsigned n = image.dim == spv::Dim2D ? 2 : 3;
      const char* kind = image.dim == spv::Dim2D   ? (image.arrayed ? "texture_2d_array" : "texture_2d")
                         : image.dim == spv::Dim3D ? "texture_3d"
                                                   : "texture_cube";
      if (image.dim != spv::Dim2D && image.dim != spv::Dim3D && image.dim != spv::DimCube)
        fail("ImageQueryLod dimension");
      if (image.dim == spv::DimCube && image.arrayed) fail("ImageQueryLod dimension");
      auto [tex, smp] = it->second;
      Value* uv = op(3);
      if (uv->type->n != n) {
        std::vector<Constant*> idx;
        for (unsigned c = 0; c < n; ++c) idx.push_back(m_.constInt(i32_, c));
        uv = b_->shuffleVector(uv, uv, m_.constAggregate(m_.vecTy(i32_, n), idx));
      }
      Value* r = m_.constUndef(rt());
      for (unsigned c = 0; c < 2; ++c)
        r = b_->insertElement(
            r,
            intrinsic(std::string(c ? "air.calculate_unclamped_lod_" : "air.calculate_clamped_lod_") + kind, f32_,
                      {tex, smp, uv, cint(0)}, AttrClass::Texture),
            cint(c));
      set(r);
      break;
    }
    case spv::OpAtomicLoad:
    case spv::OpAtomicIIncrement:
    case spv::OpAtomicIDecrement:
    case spv::OpAtomicCompareExchange:
      set(atomicExtra(in));
      break;
    case spv::OpAtomicStore:
      atomicExtra(in);
      break;
    case spv::OpAtomicAnd:
    case spv::OpAtomicOr:
    case spv::OpAtomicXor:
    case spv::OpAtomicIAdd:
    case spv::OpAtomicISub:
    case spv::OpAtomicSMin:
    case spv::OpAtomicSMax:
    case spv::OpAtomicUMin:
    case spv::OpAtomicUMax:
    case spv::OpAtomicExchange:
      set(atomic(in));
      break;
    case spv::OpImageFetch:
      set(fetch(in));
      break;
    case spv::OpMemoryBarrier:
      fail("memory barrier requires Metal 3.2 atomic fences");
    case spv::OpControlBarrier:
      controlBarrier(in);
      break;

    // ---- control flow
    case spv::OpPhi: {
      auto* phi = b_->phi(rt(), {});
      phis_.push_back({phi, in});
      set(phi);
      break;
    }
    case spv::OpBranch: b_->br(block(in.operand(0))); break;
    case spv::OpBranchConditional: b_->condBr(op(0), block(in.operand(1)), block(in.operand(2))); break;
    case spv::OpSwitch: {
      Instruction* sw = b_->switch_(op(0), block(in.operand(1)));
      for (unsigned w = 3; w + 1 < in.wc; w += 2) Builder::addCase(sw, m_.constInt(i32_, in.w[w]), block(in.w[w + 1]));
      break;
    }
    case spv::OpReturn: b_->br(exit_); break;
    case spv::OpUnreachable: b_->unreachable(); break;
    case spv::OpKill:
      intrinsic("air.discard_fragment", m_.voidTy(), {}, AttrClass::Discard);
      b_->br(exit_);
      break;
    case spv::OpDemoteToHelperInvocation:
      // The invocation becomes a helper (SPIR-V): IsHelperInvocation is true afterwards. On the GPU tested (Apple M5
      // Pro) a lane that called discard_fragment() already reports simd_is_helper_thread() == true (clean Q20); the
      // private flag keeps the SPIR-V result independent of that hardware behavior.
      if (!demoted_) {
        demoted_ = entryB_->allocaInst(i1_, 1);
        entryB_->store(m_.constNull(i1_), demoted_, 1);
      }
      b_->store(m_.constInt(i1_, 1), demoted_, 1);
      intrinsic("air.discard_fragment", m_.voidTy(), {}, AttrClass::Discard);
      break;
    case spv::OpIsHelperInvocationEXT: {
      Value* helper = intrinsic("air.simd_is_helper_thread", i1_, {}, AttrClass::Derivative);
      set(demoted_ ? b_->binop(Or, helper, b_->load(i1_, demoted_, 1)) : helper);
      break;
    }

    default: fail("opcode " + std::to_string(in.op));
  }
}

Value* Translator::extInst(const Inst& in) {
  auto a = [&](unsigned i) { return val(in.w[5 + i]); };
  auto isSplat = [&](unsigned i, float f) {
    uint32_t id = in.w[5 + i];
    if (isConstFloat(id, f)) return true;
    if (!isConstant(id)) return false;
    const Constant* c = consts_[id];
    if (c->ck == Constant::Null) return f == 0.0f && !std::signbit(f);
    if (c->ck != Constant::Aggregate) return false;
    for (Constant* e : c->ops)
      if (!isFloatBits(e, f)) return false;
    return true;
  };
  auto ints = [&](const char* name, std::vector<Value*> ops) {
    return perComponent(std::move(ops), [&](const std::vector<Value*>& s) {
      return intrinsic(name, i32_, s, AttrClass::Pure);
    });
  };
  switch (in.w[4]) {
    case GLSLstd450SAbs:
      return perComponent({a(0)}, [&](const std::vector<Value*>& s) {
        return b_->select(b_->icmp(I_SLT, s[0], cint(0)), b_->binop(Sub, cint(0), s[0]), s[0]);
      });
    case GLSLstd450SSign:
      return perComponent({a(0)}, [&](const std::vector<Value*>& s) {
        Value* positive = b_->cast(ZExt, b_->icmp(I_SGT, s[0], cint(0)), i32_);
        return b_->select(b_->icmp(I_SLT, s[0], cint(0)), cint(-1), positive);
      });
    case GLSLstd450FSign:
      return perComponent({a(0)}, [&](const std::vector<Value*>& s) {
        Value* raw = b_->cast(BitCast, s[0], i32_);
        Value* magnitude = b_->binop(And, raw, cint(0x7FFFFFFF));
        Value* active =
            b_->binop(And, b_->icmp(I_NE, magnitude, cint(0)), b_->icmp(I_ULE, magnitude, cint(0x7F800000)));
        Value* sign = b_->select(b_->icmp(I_SLT, raw, cint(0)), m_.constFloat(-1), m_.constFloat(1));
        return b_->select(active, sign, m_.constFloat(0));
      });
    case GLSLstd450Fma:
      return perComponent({a(0), a(1), a(2)}, [&](const std::vector<Value*>& s) {
        return intrinsic("air.fma.f32", f32_, s, AttrClass::Pure);
      });
    case GLSLstd450Exp:
      return fast("exp", {a(0)});
    case GLSLstd450Log:
      return fast("log", {a(0)});
    case GLSLstd450Pow:
      return fast("pow", {a(0), a(1)});
    case GLSLstd450Modf:
    case GLSLstd450ModfStruct:
      return splitFloat(in, false);
    case GLSLstd450Frexp:
    case GLSLstd450FrexpStruct:
      return splitFloat(in, true);
    case GLSLstd450PackSnorm4x8:
      return packNormalized("air.pack.snorm4x8.v4f32", a(0));
    case GLSLstd450PackUnorm4x8:
      return packNormalized("air.pack.unorm4x8.v4f32", a(0));
    case GLSLstd450PackSnorm2x16:
      return packNormalized("air.pack.snorm2x16.v2f32", a(0));
    case GLSLstd450PackUnorm2x16:
      return packNormalized("air.pack.unorm2x16.v2f32", a(0));
    case GLSLstd450UnpackSnorm4x8:
      return intrinsic("air.unpack.snorm4x8.v4f32", m_.vecTy(f32_, 4), {a(0)}, AttrClass::Pure);
    case GLSLstd450UnpackUnorm4x8:
      return intrinsic("air.unpack.unorm4x8.v4f32", m_.vecTy(f32_, 4), {a(0)}, AttrClass::Pure);
    case GLSLstd450UnpackSnorm2x16:
      return intrinsic("air.unpack.snorm2x16.v2f32", m_.vecTy(f32_, 2), {a(0)}, AttrClass::Pure);
    case GLSLstd450UnpackUnorm2x16:
      return intrinsic("air.unpack.unorm2x16.v2f32", m_.vecTy(f32_, 2), {a(0)}, AttrClass::Pure);
    case GLSLstd450FAbs: return fast("fabs", {a(0)});
    case GLSLstd450Floor: return fast("floor", {a(0)});
    case GLSLstd450Ceil: return fast("ceil", {a(0)});
    case GLSLstd450Trunc: return fast("trunc", {a(0)});
    case GLSLstd450RoundEven: return fast("rint", {a(0)});
    case GLSLstd450Fract: {
      // SPIR-V Fract is x - floor(x): 1.0 for a tiny negative x (air.fract clamps to 0x3f7fffff).
      Value* x = a(0);
      return b_->binop(FSub, x, fast("floor", {x}));
    }
    case GLSLstd450Exp2: return fast("exp2", {a(0)});
    case GLSLstd450Log2: return fast("log2", {a(0)});
    case GLSLstd450Sqrt: return fast("sqrt", {a(0)});
    case GLSLstd450InverseSqrt: return fast("rsqrt", {a(0)});
    case GLSLstd450Sin: return fast("sin", {a(0)});
    case GLSLstd450Cos: return fast("cos", {a(0)});
    case GLSLstd450FMin: case GLSLstd450NMin: return fast("fmin", {a(0), a(1)});
    case GLSLstd450FMax: case GLSLstd450NMax: return fast("fmax", {a(0), a(1)});
    case GLSLstd450FClamp: return fast("clamp", {a(0), a(1), a(2)});
    case GLSLstd450NClamp:
      if (isSplat(1, 0.0f) && isSplat(2, 1.0f)) return fast("saturate", {a(0)});
      return fast("fmin", {fast("fmax", {a(0), a(1)}), a(2)});
    case GLSLstd450SMin: return ints("air.min.s.i32", {a(0), a(1)});
    case GLSLstd450SMax: return ints("air.max.s.i32", {a(0), a(1)});
    case GLSLstd450UMin: return ints("air.min.u.i32", {a(0), a(1)});
    case GLSLstd450UMax: return ints("air.max.u.i32", {a(0), a(1)});
    case GLSLstd450UClamp:
      return ints("air.min.u.i32", {ints("air.max.u.i32", {a(0), a(1)}), a(2)});
    case GLSLstd450SClamp: return ints("air.min.s.i32", {ints("air.max.s.i32", {a(0), a(1)}), a(2)});
    case GLSLstd450Ldexp:
      // Mesa's lowering: clamp the exponent to +-254 and apply it as two exact powers of two.
      return perComponent({a(0), a(1)}, [&](const std::vector<Value*>& s) {
        Value* e = intrinsic("air.min.s.i32", i32_,
                             {intrinsic("air.max.s.i32", i32_, {s[1], cint(-254)}, AttrClass::Pure), cint(254)},
                             AttrClass::Pure);
        Value* half = b_->binop(AShr, e, cint(1));
        Value* p1 = b_->binop(Add, b_->binop(Shl, half, cint(23)), cint(0x3F800000));
        Value* p2 = b_->binop(Add, b_->binop(Shl, b_->binop(Sub, e, half), cint(23)), cint(0x3F800000));
        return b_->binop(FMul, b_->binop(FMul, s[0], b_->cast(BitCast, p1, f32_)), b_->cast(BitCast, p2, f32_));
      });
    case GLSLstd450UnpackHalf2x16: {
      Value* u = a(0);
      Value* r = m_.constUndef(m_.vecTy(f32_, 2));
      for (unsigned c = 0; c < 2; ++c) {
        Value* bits = b_->cast(Trunc, c ? b_->binop(LShr, u, cint(16)) : u, i16_);
        r = b_->insertElement(r, b_->cast(FPExt, b_->cast(BitCast, bits, f16_), f32_), cint(c));
      }
      return r;
    }
    case GLSLstd450PackHalf2x16: {
      // The default IEEE binary16 conversion of each component (SPIR-V: "converting the components ... to the 16-bit
      // OpTypeFloat representation"): fptrunc rounds to nearest even, so values from 65520 up become Inf and a NaN
      // stays a NaN (Apple's compiler emits the same fptrunc for an MSL (half) cast; tests/compute_gpu.mm compares
      // every edge value with the CPU's _Float16 conversion). Callers that need a different overflow behavior, such as
      // xenia's extended-range packer, detect the Inf themselves.
      Value* v = a(0);
      Value* r = nullptr;
      for (unsigned c = 0; c < 2; ++c) {
        Value* x = b_->extractElement(v, cint(c));
        Value* bits;
        if (options_.halfPackingRounding == HalfPackingRounding::TowardZero) {
          // Integer conversion makes the rounding policy independent of AIR's
          // fptrunc rounding, including half subnormals and finite overflow.
          Value* u = b_->cast(BitCast, x, i32_);
          Value* sign = b_->binop(And, b_->binop(LShr, u, cint(16)), cint(0x8000));
          Value* exponent = b_->binop(And, b_->binop(LShr, u, cint(23)), cint(255));
          Value* mantissa = b_->binop(And, u, cint(0x7FFFFF));
          Value* normal = b_->binop(Or, b_->binop(Shl, b_->binop(Sub, exponent, cint(112)), cint(10)),
                                   b_->binop(LShr, mantissa, cint(13)));
          Value* shift = b_->binop(Sub, cint(126), exponent);
          // Selects evaluate both inputs: never create an oversized shift.
          shift = b_->select(b_->icmp(I_ULT, shift, cint(32)), shift, cint(31));
          Value* subnormal = b_->binop(LShr, b_->binop(Or, mantissa, cint(0x800000)), shift);
          Value* small = b_->select(b_->icmp(I_UGE, exponent, cint(103)), subnormal, cint(0));
          Value* finite = b_->select(b_->icmp(I_UGE, exponent, cint(113)), normal, small);
          finite = b_->select(b_->icmp(I_UGE, exponent, cint(143)), cint(0x7BFF), finite);
          Value* special = b_->select(b_->icmp(I_NE, mantissa, cint(0)), cint(0x7E00), cint(0x7C00));
          bits = b_->binop(Or, sign, b_->select(b_->icmp(I_EQ, exponent, cint(255)), special, finite));
        } else {
          Value* h = b_->cast(FPTrunc, x, f16_);
          bits = b_->cast(ZExt, b_->cast(BitCast, h, i16_), i32_);
        }
        r = c ? b_->binop(Or, r, b_->binop(Shl, bits, cint(16))) : bits;
      }
      return r;
    }
    // The MSL forms (31 - clz(x), the sign-folded variant, ctz(x) with 0 mapped to -1) as Apple's compiler emits
    // them; clz(0) is 32, so FindUMsb(0) is -1 as GLSL.std.450 requires.
    case GLSLstd450FindUMsb:
      return findMsb(a(0), false);
    case GLSLstd450FindSMsb:
      return findMsb(a(0), true);
    case GLSLstd450FindILsb:
      return perComponent({a(0)}, [&](const std::vector<Value*>& s) {
        Value* ctz = intrinsic("air.ctz.i32", i32_, {s[0], m_.constInt(i1_, 0)}, AttrClass::Pure);
        return b_->select(b_->icmp(I_EQ, s[0], cint(0)), cint(-1), ctz);
      });
    // Lowered as MSL spells them with Apple's compiler (round, tan, ... atan2, mix, step, smoothstep): the same
    // air.<op>.f32 / air.fast_<op>.f32 intrinsics (air.mix has no fast_ spelling). Atan2 takes (y, x) as MSL does.
    case GLSLstd450Round:
      return fast("round", {a(0)});
    case GLSLstd450Tan:
      return fast("tan", {a(0)});
    case GLSLstd450Asin:
      return fast("asin", {a(0)});
    case GLSLstd450Acos:
      return fast("acos", {a(0)});
    case GLSLstd450Atan:
      return fast("atan", {a(0)});
    case GLSLstd450Atan2:
      return fast("atan2", {a(0), a(1)});
    case GLSLstd450Sinh:
      return fast("sinh", {a(0)});
    case GLSLstd450Cosh:
      return fast("cosh", {a(0)});
    case GLSLstd450Tanh:
      return fast("tanh", {a(0)});
    case GLSLstd450Asinh:
      return fast("asinh", {a(0)});
    case GLSLstd450Acosh:
      return fast("acosh", {a(0)});
    case GLSLstd450Atanh:
      return fast("atanh", {a(0)});
    case GLSLstd450Radians:
      // x * (pi / 180) with the constant rounded to float, as MSL's x * (M_PI_F / 180.0f) compiles.
      return perComponent({a(0)}, [&](const std::vector<Value*>& s) {
        return b_->binop(FMul, s[0], m_.constFPBits(f32_, 0x3C8EFA35));
      });
    case GLSLstd450Degrees:
      return perComponent({a(0)}, [&](const std::vector<Value*>& s) {
        return b_->binop(FMul, s[0], m_.constFPBits(f32_, 0x42652EE0));
      });
    case GLSLstd450FMix:
    case GLSLstd450Step:
    case GLSLstd450SmoothStep: {
      // Operands share one type in SPIR-V; a scalar next to a vector is splatted.
      unsigned count = in.w[4] == GLSLstd450Step ? 2 : 3;
      std::vector<Value*> ops;
      unsigned n = 1;
      for (unsigned k = 0; k < count; ++k) {
        ops.push_back(a(k));
        if (ops.back()->type->kind == Type::Vector) n = ops.back()->type->n;
      }
      if (n > 1)
        for (Value*& o : ops)
          if (o->type->kind != Type::Vector) o = splat(o, n);
      if (in.w[4] == GLSLstd450FMix)
        // MSL mix(x, y, a); SPIR-V defines x * (1 - a) + y * a, which air.mix computes for finite operands.
        return perComponent(
            ops, [&](const std::vector<Value*>& s) { return intrinsic("air.mix.f32", f32_, s, AttrClass::Pure); });
      if (in.w[4] == GLSLstd450Step)
        // SPIR-V: 0 if x < edge, otherwise 1, so a NaN operand gives 1 (unordered >=). MSL step() compiles to an
        // ordered x >= edge and gives 0 there.
        return perComponent(ops, [&](const std::vector<Value*>& s) {
          return intrinsic("air.convert.f.f32.u.i1", f32_, {b_->fcmp(F_UGE, s[1], s[0])}, AttrClass::Pure);
        });
      // clamp((x - e0) / (e1 - e0), 0, 1), then t * t * (3 - 2 * t); the multiply-add contracts in compute only.
      return perComponent(ops, [&](const std::vector<Value*>& s) {
        Value* num = b_->binop(FSub, s[2], s[0]);
        Value* den = b_->binop(FSub, s[1], s[0]);
        // Fast math (graphics) turns the division into a multiply by the reciprocal.
        Value* ratio =
            compute() ? b_->binop(FDiv, num, den) : b_->binop(FMul, num, b_->binop(FDiv, m_.constFloat(1), den));
        Value* t = fast("clamp", {ratio, m_.constFloat(0), m_.constFloat(1)});
        Value* sq = b_->binop(FMul, t, t);
        Value* poly =
            compute() ? intrinsic("llvm.fmuladd.f32", f32_, {t, m_.constFloat(-2), m_.constFloat(3)}, AttrClass::Pure)
                      : b_->binop(FSub, m_.constFloat(3), b_->binop(FMul, t, m_.constFloat(2)));
        return b_->binop(FMul, sq, poly);
      });
    }
    case GLSLstd450Length:
    case GLSLstd450Distance:
    case GLSLstd450Cross:
    case GLSLstd450Normalize:
    case GLSLstd450FaceForward:
    case GLSLstd450Reflect:
    case GLSLstd450Refract:
      return geometric(in);
    default: fail("GLSL.std.450 instruction " + std::to_string(in.w[4]));
  }
}

// Length, Distance, Cross, Normalize, FaceForward, Reflect and Refract as Apple's compiler expands the MSL functions:
// air.dot.v<N>f32 for the dot products and sqrt/rsqrt (air.fast_ in graphics) for the roots. With fast math off
// (compute) the expansion keeps its scaling for overflow/underflow in length and normalize and contracts a*b+c into
// llvm.fmuladd; Apple's compiler branches there, this selects between the same values (all of them are pure). Under
// fast math (graphics) Reflect and Cross are still contracted (matches bit for bit on the GPU tested); Refract's
// compiled expression differs from the IR in the last bits, so it is written as the IR spells it. A scalar operand
// has no MSL spelling: it uses the same formulas with the dot product x * y.
Value* Translator::geometric(const Inst& in) {
  const uint32_t op = in.w[4];
  auto a = [&](unsigned i) { return val(in.w[5 + i]); };
  const bool precise = compute();
  Value* x = a(0);
  const unsigned n = x->type->kind == Type::Vector ? x->type->n : 1;
  auto lanes = [&](Value* v) {
    std::vector<Value*> r;
    for (unsigned c = 0; c < n; ++c) r.push_back(n > 1 ? b_->extractElement(v, cint(c)) : v);
    return r;
  };
  auto pack = [&](const std::vector<Value*>& s) -> Value* {
    if (n == 1) return s[0];
    Value* r = m_.constUndef(m_.vecTy(f32_, n));
    for (unsigned c = 0; c < n; ++c) r = b_->insertElement(r, s[c], cint(c));
    return r;
  };
  auto each = [&](const std::function<Value*(unsigned)>& f) {
    std::vector<Value*> s;
    for (unsigned c = 0; c < n; ++c) s.push_back(f(c));
    return pack(s);
  };
  auto mul = [&](Value* p, Value* q) { return b_->binop(FMul, p, q); };
  auto fmad = [&](Value* p, Value* q, Value* r) {
    return intrinsic("llvm.fmuladd.f32", f32_, {p, q, r}, AttrClass::Pure);
  };
  auto dot = [&](Value* p, Value* q) -> Value* {
    if (n == 1) return mul(p, q);
    return intrinsic("air.dot.v" + std::to_string(n) + "f32", f32_, {p, q}, AttrClass::Pure);
  };
  auto isInf = [&](Value* f) {
    return b_->icmp(I_EQ, b_->binop(And, b_->cast(BitCast, f, i32_), cint(0x7FFFFFFF)), cint(0x7F800000));
  };
  auto scale = [&](const std::vector<Value*>& v, float k) {
    std::vector<Value*> s;
    for (Value* e : v) s.push_back(mul(e, m_.constFloat(k)));
    return s;
  };
  auto length = [&](const std::vector<Value*>& v) -> Value* {
    if (n == 1) return fast("fabs", {v[0]});
    Value* d = dot(pack(v), pack(v));
    if (!precise) return fast("sqrt", {d});
    auto scaledRoot = [&](float k) {
      Value* p = pack(scale(v, k));
      return fast("sqrt", {dot(p, p)});
    };
    Value* big = mul(scaledRoot(std::ldexp(1.0f, -66)), m_.constFloat(std::ldexp(1.0f, 66)));
    Value* small = mul(scaledRoot(std::ldexp(1.0f, 64)), m_.constFloat(std::ldexp(1.0f, -64)));
    return b_->select(isInf(d), big,
                      b_->select(b_->fcmp(F_OLT, d, m_.constFloat(std::ldexp(1.0f, -103))), small, fast("sqrt", {d})));
  };
  switch (op) {
    case GLSLstd450Length:
      return length(lanes(x));
    case GLSLstd450Distance: {
      Value* y = a(1);
      std::vector<Value*> p = lanes(x), q = lanes(y), d;
      for (unsigned c = 0; c < n; ++c) d.push_back(b_->binop(FSub, p[c], q[c]));
      return length(d);
    }
    case GLSLstd450Normalize: {
      std::vector<Value*> v = lanes(x);
      Value* d = dot(x, x);
      if (!precise) {
        Value* r = fast("rsqrt", {d});
        return each([&](unsigned c) { return mul(v[c], r); });
      }
      // Infinite component: NaN. Dot overflow: rescale by 2^-66; dot underflow: by 2^64 (an all-zero vector stays
      // that scaled zero); then multiply by rsqrt of the rescaled dot.
      Value* anyInf = isInf(v[0]);
      for (unsigned c = 1; c < n; ++c) anyInf = b_->binop(Or, anyInf, isInf(v[c]));
      std::vector<Value*> y = scale(v, std::ldexp(1.0f, -66)), z = scale(v, std::ldexp(1.0f, 64));
      Value *d1 = dot(pack(y), pack(y)), *d2 = dot(pack(z), pack(z));
      Value* over = isInf(d);
      Value* under = b_->fcmp(F_OLT, d, m_.constFloat(std::ldexp(1.0f, -103)));
      Value* dd = b_->select(over, d1, b_->select(under, d2, d));
      Value* r = fast("rsqrt", {dd});
      Value* zeros = b_->binop(And, under, b_->fcmp(F_OEQ, d2, m_.constFloat(0)));
      Value* nan = m_.constFPBits(f32_, 0x7FC00000);
      return each([&](unsigned c) {
        Value* vc = b_->select(over, y[c], b_->select(under, z[c], v[c]));
        return b_->select(anyInf, nan, b_->select(zeros, z[c], mul(vc, r)));
      });
    }
    case GLSLstd450FaceForward: {
      // N if dot(Nref, I) < 0, else -N.
      std::vector<Value*> v = lanes(x);
      Value* neg = b_->fcmp(F_OLT, dot(a(2), a(1)), m_.constFloat(0));
      return each([&](unsigned c) { return b_->select(neg, v[c], b_->fneg(v[c])); });
    }
    case GLSLstd450Reflect: {
      // I - 2 * dot(N, I) * N.
      std::vector<Value*> i = lanes(x), nn = lanes(a(1));
      Value* t = mul(dot(a(1), x), m_.constFloat(2));
      return each([&](unsigned c) { return fmad(b_->fneg(t), nn[c], i[c]); });
    }
    case GLSLstd450Refract: {
      // k = 1 - eta^2 * (1 - dot(N, I)^2); 0 if k < 0, else eta * I - (eta * dot(N, I) + sqrt(k)) * N.
      std::vector<Value*> i = lanes(x), nn = lanes(a(1));
      Value *eta = a(2), *d = dot(a(1), x), *k;
      Value* one = m_.constFloat(1);
      if (precise) {
        k = fmad(mul(eta, b_->fneg(eta)), fmad(b_->fneg(d), d, one), one);
      } else {
        k = b_->binop(FSub, one, mul(mul(eta, eta), b_->binop(FSub, one, mul(d, d))));
      }
      Value* neg = b_->fcmp(F_OLT, k, m_.constFloat(0));
      Value* root = fast("sqrt", {k});
      Value* m = precise ? fmad(eta, d, root) : b_->binop(FAdd, root, mul(d, eta));
      return each([&](unsigned c) {
        Value* r = precise ? fmad(eta, i[c], mul(nn[c], b_->fneg(m))) : b_->binop(FSub, mul(eta, i[c]), mul(m, nn[c]));
        return b_->select(neg, m_.constFloat(0), r);
      });
    }
    case GLSLstd450Cross: {
      if (n != 3) fail("Cross on other than a 3-component vector");
      std::vector<Value*> p = lanes(x), q = lanes(a(1));
      return each([&](unsigned c) -> Value* {
        unsigned j = (c + 1) % 3, k = (c + 2) % 3;  // p[j]*q[k] - q[j]*p[k]
        return fmad(p[j], q[k], mul(p[k], b_->fneg(q[j])));
      });
    }
  }
  fail("GLSL.std.450 geometric instruction " + std::to_string(op));
}

Value* Translator::packNormalized(const char* name, Value* x) {
  // Normalized packing leaves NaNs unspecified; use the D3D conversion choice of zero.
  Value* value = perComponent({x}, [&](const std::vector<Value*>& s) {
    return b_->select(b_->fcmp(F_UNO, s[0], s[0]), m_.constFloat(0), s[0]);
  });
  return intrinsic(name, i32_, {value}, AttrClass::Pure);
}

// Modf preserves the input sign on a zero fraction (including infinities). Frexp normalizes subnormals
// exactly before reading the exponent; special values keep their bits and have a deterministic zero exponent.
Value* Translator::splitFloat(const Inst& in, bool frexp) {
  Value* x = val(in.w[5]);
  auto part = [&](bool second) {
    return perComponent({x}, [&](const std::vector<Value*>& s) -> Value* {
      Value* raw = b_->cast(BitCast, s[0], i32_);
      Value* magnitude = b_->binop(And, raw, cint(0x7FFFFFFF));
      if (!frexp) {
        Value* integral = intrinsic("air.trunc.f32", f32_, {s[0]}, AttrClass::Pure);
        if (second) return integral;
        Value* frac = b_->cast(BitCast, b_->binop(FSub, s[0], integral), i32_);
        Value* isInf = b_->icmp(I_EQ, magnitude, cint(0x7F800000));
        Value* absFrac = b_->binop(And, frac, cint(0x7FFFFFFF));
        absFrac = b_->select(b_->icmp(I_ULT, magnitude, cint(0x3F800000)), magnitude, absFrac);
        Value* sign = b_->binop(And, raw, cint(0x80000000));
        return b_->cast(BitCast, b_->binop(Or, b_->select(isInf, cint(0), absFrac), sign), f32_);
      }
      Value* tiny = b_->icmp(I_ULT, magnitude, cint(0x00800000));
      Value* leading = intrinsic("air.clz.i32", i32_, {magnitude, m_.constInt(i1_, 0)}, AttrClass::Pure);
      Value* shift = b_->binop(And, b_->binop(Sub, leading, cint(8)), cint(31));
      Value* mantissa = b_->binop(And, b_->binop(Shl, magnitude, shift), cint(0x007FFFFF));
      Value* sign = b_->binop(And, raw, cint(0x80000000));
      Value* normalized = b_->select(tiny, b_->binop(Or, b_->binop(Or, sign, mantissa), cint(0x00800000)), raw);
      Value* special = b_->binop(Or, b_->icmp(I_EQ, magnitude, cint(0)), b_->icmp(I_UGE, magnitude, cint(0x7F800000)));
      if (second) {
        Value* exp = b_->binop(And, b_->binop(LShr, normalized, cint(23)), cint(255));
        exp = b_->binop(Sub, b_->binop(Sub, exp, cint(126)), b_->select(tiny, shift, cint(0)));
        return b_->select(special, cint(0), exp);
      }
      Value* fraction = b_->binop(Or, b_->binop(And, normalized, cint(0x807FFFFF)), cint(0x3F000000));
      return b_->cast(BitCast, b_->select(special, raw, fraction), f32_);
    });
  };
  Value *fraction = part(false), *whole = part(true);
  bool structure = in.w[4] == GLSLstd450ModfStruct || in.w[4] == GLSLstd450FrexpStruct;
  if (!structure) {
    store(ptrs_[in.w[6]], whole);
    return fraction;
  }
  Value* r = b_->insertValue(m_.constUndef(llvmType(in.operand(0))), fraction, {0});
  return b_->insertValue(r, whole, {1});
}

// FindUMsb / FindSMsb: the index of the most significant set bit, or of the most significant bit differing from
// the sign for a negative signed value (so FindSMsb(-1) is -1 as well).
Value* Translator::findMsb(Value* x, bool sign) {
  return perComponent({x}, [&](const std::vector<Value*>& s) {
    Value* v = s[0];
    if (sign) v = b_->binop(Xor, b_->binop(AShr, v, cint(31)), v);
    Value* clz = intrinsic("air.clz.i32", i32_, {v, m_.constInt(i1_, 0)}, AttrClass::Pure);
    return b_->binop(Sub, cint(31), clz);
  });
}

// Queries use the same texture handle as reads/samples; array layer counts are not mipmapped.
Value* Translator::imageQuery(const Inst& in) {
  auto it = loadedImageType_.find(in.operand(2));
  if (it == loadedImageType_.end()) fail("query from something other than an image");
  const SType& image = type(it->second);
  std::string suffix;
  if (compute()) {
    if (image.dim != spv::Dim2D || image.arrayed) fail("compute image query dimension");
    suffix = image.multisampled ? "texture_2d_ms" : "texture_2d";
  } else {
    if (image.multisampled) fail("multisampled graphics image query");
    if (image.dim == spv::Dim2D)
      suffix = "texture_2d_array";
    else if (image.dim == spv::DimCube)
      suffix = "texture_cube_array";
    else if (image.dim == spv::Dim3D)
      suffix = "texture_3d";
    else
      fail("image query dimension");
  }
  Value* texture = val(in.operand(2));
  auto query = [&](const char* what, bool lod) {
    std::vector<Value*> args = {texture};
    if (lod && !image.multisampled) args.push_back(in.op == spv::OpImageQuerySizeLod ? val(in.operand(3)) : cint(0));
    return intrinsic(std::string("air.get_") + what + "_" + suffix, i32_, args, AttrClass::TextureRead);
  };
  if (in.op == spv::OpImageQueryLevels) {
    if (image.multisampled) fail("mip levels of multisampled image");
    return query("num_mip_levels", false);
  }
  if (in.op == spv::OpImageQuerySamples) {
    if (!image.multisampled) fail("sample count of non-multisampled image");
    return query("num_samples", false);
  }
  Type* result = llvmType(in.operand(0));
  unsigned dimensions = image.dim == spv::Dim3D ? 3 : 2;
  Value* r = m_.constUndef(result);
  const char* names[] = {"width", "height", "depth"};
  for (unsigned k = 0; k < dimensions; ++k) r = b_->insertElement(r, query(names[k], true), cint(k));
  if (image.arrayed) r = b_->insertElement(r, query("array_size", false), cint(dimensions));
  return r;
}

// The remaining 32-bit atomic forms. Metal 3.1 only exposes relaxed memory ordering. A strong compare/exchange
// retries a weak failure only while the observed value still equals the comparator, so spurious failures cannot leak.
Value* Translator::atomicExtra(const Inst& in) {
  bool isStore = in.op == spv::OpAtomicStore;
  unsigned base = isStore ? 0 : 2;
  const Pointer& p = ptrs_[in.operand(base)];
  bool local = compute() && p.kind == Pointer::Private && p.addrSpace == 3;
  if (!local && (p.kind != Pointer::Buffer || p.addrSpace != 1)) fail("atomic on non-device memory");
  uint32_t scope = constU32(in.operand(base + 1));
  if (scope != spv::ScopeDevice && scope != spv::ScopeWorkgroup && scope != spv::ScopeSubgroup &&
      scope != spv::ScopeInvocation)
    fail("atomic scope");
  uint32_t semantics = constU32(in.operand(base + 2));
  if (semantics & ~uint32_t(spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsWorkgroupMemoryMask |
                            spv::MemorySemanticsCrossWorkgroupMemoryMask | spv::MemorySemanticsVolatileMask))
    fail("atomic memory ordering requires Metal 3.2");
  Value* ptr = local ? p.ptr : bufferAddress(p, i32_);
  Value* isVolatile = m_.constInt(i1_, bool(semantics & spv::MemorySemanticsVolatileMask));
  std::string prefix = std::string("air.atomic.") + (local ? "local." : "global.");
  if (in.op == spv::OpAtomicLoad)
    return intrinsic(prefix + "load.i32", i32_, {ptr, cint(0), cint(local ? 1 : 2), isVolatile}, AttrClass::Atomic);
  if (isStore)
    return intrinsic(prefix + "store.i32", m_.voidTy(),
                     {ptr, val(in.operand(3)), cint(0), cint(local ? 1 : 2), isVolatile}, AttrClass::Atomic);
  if (in.op == spv::OpAtomicIIncrement || in.op == spv::OpAtomicIDecrement)
    return intrinsic(prefix + (in.op == spv::OpAtomicIIncrement ? "add.u.i32" : "sub.u.i32"), i32_,
                     {ptr, cint(1), cint(0), cint(local ? 1 : 2), m_.constInt(i1_, 1)}, AttrClass::Atomic);
  uint32_t failure = constU32(in.operand(5));
  if (failure & ~uint32_t(spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsWorkgroupMemoryMask |
                          spv::MemorySemanticsCrossWorkgroupMemoryMask | spv::MemorySemanticsVolatileMask))
    fail("atomic failure ordering requires Metal 3.2");
  isVolatile = m_.constInt(i1_, bool((semantics | failure) & spv::MemorySemanticsVolatileMask));
  std::string name = std::string("s2a.cmpxchg.") + (local ? "local" : "global");
  Function* helper = nullptr;
  if (auto it = decls_.find(name); it != decls_.end())
    helper = it->second;
  else {
    Function* weak = declare(prefix + "cmpxchg.weak.i32", i32_,
                             {ptr->type, m_.ptrTy(i32_), i32_, i32_, i32_, i32_, i1_}, AttrClass::Atomic);
    helper = declare(name, i32_, {ptr->type, i32_, i32_, i1_}, AttrClass::Atomic);
    helper->linkage = Link_Internal;
    BasicBlock *entry = m_.addBlock(helper), *retry = m_.addBlock(helper), *done = m_.addBlock(helper);
    Builder h(m_, entry);
    Value* expected = h.allocaInst(i32_, 4);
    h.store(helper->args[1], expected, 4);
    h.br(retry);
    h.setInsertBlock(retry);
    Value* success = h.call(
        weak, {helper->args[0], expected, helper->args[2], cint(0), cint(0), cint(local ? 1 : 2), helper->args[3]});
    Value* observed = h.load(i32_, expected, 4);
    Value* stop = h.icmp(I_NE, success, cint(0));
    stop = h.binop(Or, stop, h.icmp(I_NE, observed, helper->args[1]));
    h.condBr(stop, done, retry);
    h.setInsertBlock(done);
    h.ret(observed);
#ifndef NDEBUG
    if (std::string problem = verifyFunction(helper); !problem.empty()) fail("atomic helper: " + problem);
    if (std::string problem = verifyDominance(helper); !problem.empty()) fail("atomic helper: " + problem);
#endif
  }
  return b_->call(helper, {ptr, val(in.operand(7)), val(in.operand(6)), isVolatile});
}

// Read-modify-write atomics on device or threadgroup memory. Guest shaders use And/Or on the shared memory
// buffer and Add on the occlusion counter; compute shaders get the 32-bit integer operations.
Value* Translator::atomic(const Inst& in) {
  uint32_t scope = constU32(in.operand(3)), semantics = constU32(in.operand(4));
  if (scope != spv::ScopeDevice && scope != spv::ScopeWorkgroup && scope != spv::ScopeSubgroup &&
      scope != spv::ScopeInvocation)
    fail("atomic scope");
  if (semantics & ~uint32_t(spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsWorkgroupMemoryMask |
                            spv::MemorySemanticsCrossWorkgroupMemoryMask | spv::MemorySemanticsVolatileMask))
    fail("atomic memory ordering requires Metal 3.2");
  const Pointer& p = ptrs_[in.operand(2)];
  bool local = compute() && p.kind == Pointer::Private && p.addrSpace == 3;
  if (!local && (p.kind != Pointer::Buffer || p.addrSpace != 1)) fail("atomic on non-device memory");
  const char* name = nullptr;
  char sign = compute() ? 'u' : 's';
  switch (in.op) {
    case spv::OpAtomicAnd:
      name = "and";
      break;
    case spv::OpAtomicOr:
      name = "or";
      break;
    case spv::OpAtomicIAdd:
      // The ZPD counter's increments in pixel shaders: MSL atomic_uint.
      name = "add";
      sign = 'u';
      break;
    default:
      if (!compute() && in.op != spv::OpAtomicSMin && in.op != spv::OpAtomicSMax && in.op != spv::OpAtomicUMin &&
          in.op != spv::OpAtomicUMax)
        fail("atomic operation " + std::to_string(in.op));
      switch (in.op) {
        case spv::OpAtomicXor:
          name = "xor";
          break;
        case spv::OpAtomicISub:
          name = "sub";
          break;
        case spv::OpAtomicSMin:
          name = "min";
          sign = 's';
          break;
        case spv::OpAtomicSMax:
          name = "max";
          sign = 's';
          break;
        case spv::OpAtomicUMin:
          name = "min";
          break;
        case spv::OpAtomicUMax:
          name = "max";
          break;
        case spv::OpAtomicExchange:
          name = "xchg";
          sign = 0;
          break;
        default:
          fail("atomic operation " + std::to_string(in.op));
      }
  }
  std::string fn = std::string("air.atomic.") + (local ? "local." : "global.") + name;
  if (sign) fn += std::string(".") + sign;
  fn += ".i32";
  Value* ptr = local ? p.ptr : bufferAddress(p, i32_);
  return intrinsic(fn, i32_, {ptr, val(in.operand(5)), cint(0), cint(local ? 1 : 2), m_.constInt(i1_, 1)},
                   AttrClass::Atomic);
}

// OpControlBarrier: the memory classes in its semantics select the mem_flags of threadgroup_barrier.
void Translator::controlBarrier(const Inst& in) {
  if (!compute()) fail("barrier outside compute");
  uint32_t execution = constU32(in.operand(0)), semantics = constU32(in.operand(2));
  uint32_t flags = 0;
  if (semantics & spv::MemorySemanticsWorkgroupMemoryMask) flags |= 2;
  if (semantics & (spv::MemorySemanticsUniformMemoryMask | spv::MemorySemanticsCrossWorkgroupMemoryMask)) flags |= 1;
  if (semantics & spv::MemorySemanticsImageMemoryMask) flags |= 4;
  if (semantics & ~uint32_t(spv::MemorySemanticsWorkgroupMemoryMask | spv::MemorySemanticsUniformMemoryMask |
                            spv::MemorySemanticsCrossWorkgroupMemoryMask | spv::MemorySemanticsImageMemoryMask |
                            spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsAcquireMask |
                            spv::MemorySemanticsReleaseMask | spv::MemorySemanticsSequentiallyConsistentMask))
    fail("barrier semantics " + std::to_string(semantics));
  if (execution != spv::ScopeWorkgroup && execution != spv::ScopeSubgroup) fail("barrier scope");
  intrinsic(execution == spv::ScopeWorkgroup ? "air.wg.barrier" : "air.simdgroup.barrier", m_.voidTy(),
            {cint(flags), cint(1)}, AttrClass::Barrier);
}

// OpImageFetch of a sampled 2D image, optionally multisampled: texture.read(coordinate, lod or sample).
Value* Translator::fetch(const Inst& in) {
  if (!compute()) fail("OpImageFetch outside compute");
  auto it = loadedImageType_.find(in.operand(2));
  if (it == loadedImageType_.end()) fail("fetch from something other than a loaded image variable");
  const SType& image = type(it->second);
  const SType& texel = type(image.elem);
  uint32_t mask = in.wc > 5 ? in.operand(4) : 0;
  if (mask & ~uint32_t(spv::ImageOperandsLodMask | spv::ImageOperandsSampleMask)) fail("fetch operands");
  if (bool(mask & spv::ImageOperandsSampleMask) != image.multisampled) fail("fetch operand does not match the image");
  Value* index = mask ? val(in.w[6]) : cint(0);
  bool isFloat = texel.op == spv::OpTypeFloat;
  Type* vec = m_.vecTy(isFloat ? f32_ : i32_, 4);
  std::string name = std::string("air.read_texture_2d") + (image.multisampled ? "_ms" : "") + "." +
                     (isFloat          ? ""
                      : texel.isSigned ? "s."
                                       : "u.") +
                     (isFloat ? "v4f32" : "v4i32");
  // The last immediate is the texture's declared access: sample 0, read 1 (a multisampled texture is read).
  Value* r = intrinsic(name, m_.literalStructTy({vec, i8_}),
                       {val(in.operand(2)), val(in.operand(3)), index, cint(image.multisampled ? 1 : 0)},
                       AttrClass::TextureRead);
  return b_->extractValue(r, {0});
}

// The array layer of an image sample: SPIR-V (and Vulkan) take clamp(RoundEven(coordinate), 0, layers - 1). rint is the
// round-half-even; air.convert.u.i32.f.f32 is the conversion Apple's compiler writes for an MSL float -> uint cast,
// which saturates (negative and NaN become 0, too large becomes 0xffffffff) where fptoui would be poison. The upper
// clamp to the last layer is the sampler's (checked by tests/tex_gpu.mm with 1e20).
Value* Translator::arrayLayer(Value* coordinate) {
  return intrinsic("air.convert.u.i32.f.f32", i32_, {fast("rint", {coordinate})}, AttrClass::Pure);
}

Value* Translator::sample(const Inst& in) {
  auto it = sampledImages_.find(in.operand(2));
  if (it == sampledImages_.end()) fail("sample without OpSampledImage");
  auto [tex, smp] = it->second;
  const SType& image = type(sampledImageType_.at(in.operand(2)));
  Value* coord = val(in.operand(3));
  uint32_t mask = in.wc > 5 ? in.operand(4) : 0;
  unsigned w = 6;
  bool lod = false;
  Value* value = m_.constFloat(0.0f);
  Value *dx = nullptr, *dy = nullptr;
  if (mask & spv::ImageOperandsBiasMask) value = val(in.w[w++]);
  if (mask & spv::ImageOperandsLodMask) {
    value = val(in.w[w++]);
    lod = true;
  }
  if (mask & spv::ImageOperandsGradMask) {
    dx = val(in.w[w++]);
    dy = val(in.w[w++]);
  }
  if (mask & ~uint32_t(spv::ImageOperandsBiasMask | spv::ImageOperandsLodMask | spv::ImageOperandsGradMask))
    fail("image operands " + std::to_string(mask));
  Type* resultTy = m_.literalStructTy({v4f32_, i8_});
  Value* zero = m_.constFloat(0.0f);
  Value* t = m_.constInt(i1_, 1);
  Value* lodFlag = m_.constInt(i1_, lod ? 1 : 0);
  Value* r = nullptr;
  auto lanes = [&](unsigned n) {  // the first n components of the coordinate
    std::vector<Constant*> idx;
    for (unsigned c = 0; c < n; ++c) idx.push_back(m_.constInt(i32_, c));
    return b_->shuffleVector(coord, coord, m_.constAggregate(m_.vecTy(i32_, n), idx));
  };
  switch (image.dim) {
    case spv::Dim2D: {
      Value* uv = lanes(2);
      Value* offset = m_.constNull(m_.vecTy(i32_, 2));
      if (image.arrayed) {
        Value* slice = arrayLayer(b_->extractElement(coord, cint(2)));
        if (dx)
          r = intrinsic("air.sample_texture_2d_array_grad.v4f32", resultTy,
                        {tex, smp, uv, slice, dx, dy, zero, t, offset, cint(0)}, AttrClass::TextureRead);
        else
          r = intrinsic("air.sample_texture_2d_array.v4f32", resultTy,
                        {tex, smp, uv, slice, t, offset, lodFlag, value, zero, cint(0)}, AttrClass::Texture);
      } else if (dx) {
        r = intrinsic("air.sample_texture_2d_grad.v4f32", resultTy, {tex, smp, uv, dx, dy, zero, t, offset, cint(0)},
                      AttrClass::TextureRead);
      } else {
        r = intrinsic("air.sample_texture_2d.v4f32", resultTy, {tex, smp, uv, t, offset, lodFlag, value, zero, cint(0)},
                      AttrClass::Texture);
      }
      break;
    }
    case spv::DimCube: {
      Value* dir = lanes(3);
      if (image.arrayed) {
        Value* slice = arrayLayer(b_->extractElement(coord, cint(3)));
        if (dx)
          r = intrinsic("air.sample_texture_cube_array_grad.v4f32", resultTy,
                        {tex, smp, dir, slice, dx, dy, zero, cint(0)}, AttrClass::TextureRead);
        else
          r = intrinsic("air.sample_texture_cube_array.v4f32", resultTy,
                        {tex, smp, dir, slice, lodFlag, value, zero, cint(0)}, AttrClass::Texture);
      } else if (dx) {
        r = intrinsic("air.sample_texture_cube_grad.v4f32", resultTy, {tex, smp, dir, dx, dy, zero, cint(0)},
                      AttrClass::TextureRead);
      } else {
        r = intrinsic("air.sample_texture_cube.v4f32", resultTy, {tex, smp, dir, lodFlag, value, zero, cint(0)},
                      AttrClass::Texture);
      }
      break;
    }
    case spv::Dim3D:
      if (dx)
        r = intrinsic("air.sample_texture_3d_grad.v4f32", resultTy,
                      {tex, smp, coord, dx, dy, zero, t, m_.constNull(m_.vecTy(i32_, 3)), cint(0)},
                      AttrClass::TextureRead);
      else
        r = intrinsic("air.sample_texture_3d.v4f32", resultTy,
                      {tex, smp, coord, t, m_.constNull(m_.vecTy(i32_, 3)), lodFlag, value, zero, cint(0)},
                      AttrClass::Texture);
      break;
    default: fail("image dimension");
  }
  return b_->extractValue(r, {0});
}

// ---------------------------------------------------------------- driver

std::vector<uint8_t> Translator::run() {
  // The triple Apple's compiler writes for the deployment target, e.g. `air64_v26-apple-macosx14.0.0` (see
  // mtlb::triple). An absent triple is rejected (metal-as then assumes AIR 1.8).
  char triple[64];
  m_.triple = mtlb::triple(platform_, target_.os_version, triple, sizeof triple);
  m_.datalayout = kDataLayout;
  m_.sourceFileName = "main";
  i1_ = m_.intTy(1);
  i8_ = m_.intTy(8);
  i16_ = m_.intTy(16);
  i32_ = m_.intTy(32);
  i64_ = m_.intTy(64);
  f16_ = m_.halfTy();
  f32_ = m_.floatTy();
  v4f32_ = m_.vecTy(f32_, 4);
  tlabTy_ = m_.structTy("struct.ArgumentBuffer");
  resHeapTy_ = m_.structTy("struct.ResourceHeap");
  smpHeapTy_ = m_.structTy("struct.SamplerHeap");
  samplerTy_ = m_.structTy("struct._sampler_t");
  // The handle type follows the image type's Dim and Arrayed, so it matches the view type the host binds
  // (MTLTextureType2D/2DArray/Cube/CubeArray/3D). Apple's compiler writes these struct names for the MSL texture types.
  textureTypes_[spv::Dim2D] = m_.structTy("struct._texture_2d_t");
  textureTypes_[spv::Dim2D | kArrayedKey] = m_.structTy("struct._texture_2d_array_t");
  textureTypes_[spv::DimCube] = m_.structTy("struct._texture_cube_t");
  textureTypes_[spv::DimCube | kArrayedKey] = m_.structTy("struct._texture_cube_array_t");
  textureTypes_[spv::Dim3D] = m_.structTy("struct._texture_3d_t");

  StageTimer timer;
  // Debugging aid: S2A_DUMP_DIR=<dir> writes the module's bitcode after every pass (NN_pass.bc, for llvm-dis).
  const char* dumpDir = std::getenv("S2A_DUMP_DIR");
  int dumpIndex = 0;
  auto dump = [&](const char* pass) {
    if (!dumpDir) return;
    if (std::string problem = verifyFunction(main_); !problem.empty())
      fprintf(stderr, "VERIFY after %s: %s\n", pass, problem.c_str());
    if (std::string problem = verifyDominance(main_); !problem.empty())
      fprintf(stderr, "DOMINANCE after %s: %s\n", pass, problem.c_str());
    std::vector<uint8_t> bc = writeBitcode(m_);
    // The writer caches value ids (id + 1) in the scratch tags. The passes' dense numbering tests `tag - base < n` and
    // would take such a stale tag for its own, so move the tag counter past every id the writer handed out (there
    // are fewer ids than bytes of bitcode).
    m_.allocTags(uint32_t(bc.size()) + 64);
    char path[1024];
    snprintf(path, sizeof path, "%s/%02d_%s.bc", dumpDir, dumpIndex++, pass);
    if (FILE* f = fopen(path, "wb")) fwrite(bc.data(), 1, bc.size(), f), fclose(f);
  };
  decode();
  buildSignature();
  buildEntry();
  translateFunction();
  buildExit();
  timer.lap(0);
  dump("translate");
  promoteAllocas(m_, main_);
  dump("promoteAllocas");
  timer.lap(1);
  scalarizeFunction(m_, main_);
  dump("scalarize");
  timer.lap(2);
  cseFunction(m_, main_);
  dump("cse");
  timer.lap(3);
  simplifyInstructions(m_, main_);
  dump("simplifyInstructions");
  cleanupFunction(m_, main_);
  dump("cleanup");
  // Thread the Xenos dispatch loop's known jumps (Options::jumpThread), then clean up.
  if (options_.jumpThread && jumpThread(m_, main_)) cleanupFunction(m_, main_);
  // Bypass empty blocks and if-convert small arms, then let cleanup merge and resolve what that leaves in a line.
  for (int round = 0; round < 8; ++round) {
    bool changed = simplifyCfg(m_, main_);
    dump("simplifyCfg");
    changed |= simplifyInstructions(m_, main_);
    dump("simplifyInstructions");
    if (!changed) break;
    cleanupFunction(m_, main_);
    dump("cleanup");
  }
  timer.lap(4);
#if !defined(NDEBUG) || defined(SPIRV2AIR_ENABLE_VERIFY)
  if (std::string problem = verifyFunction(main_); !problem.empty()) fail("malformed IR after cleanup: " + problem);
  if (std::string problem = verifyDominance(main_); !problem.empty()) fail("malformed IR after cleanup: " + problem);
#endif
  finalizeMetadata();
  dump("final");

  std::vector<uint8_t> bitcode = writeBitcode(m_);
  timer.lap(5);
  mtlb::Function fn;
  fn.name = "main";
  fn.type = compute()                               ? mtlb::FunctionType::Kernel
            : model_ == spv::ExecutionModelFragment ? mtlb::FunctionType::Fragment
                                                    : mtlb::FunctionType::Vertex;
  if (tese()) fn.tessellation = mtlb::tessellation_tag(patchDomain_ == PatchDomain::Quad, controlPoints_);
  fn.air = airVersion_;
  fn.constants = fcEntries_.data();
  fn.num_constants = uint32_t(fcEntries_.size());
  fn.bitcode = bitcode.data();
  fn.bitcode_size = bitcode.size();
  std::vector<uint8_t> out(mtlb::library_size(&fn, 1));
  out.resize(mtlb::write(out.data(), target_, &fn, 1));
  timer.lap(6);
  return out;
}

}  // namespace

Result translate(const uint32_t* words, size_t word_count, const Options& options) {
  Result r;
  try {
    r.metallib = Translator(words, word_count, options).run();
  } catch (const std::exception& e) {
    r.error = e.what();
  }
  return r;
}

}  // namespace spirv2air
