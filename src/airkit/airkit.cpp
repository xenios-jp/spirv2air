// airkit: Metal kernel construction helpers; see airkit.h.
#include "airkit/airkit.h"

#include <stdexcept>

#include "metallib/metallib.h"

namespace airkit {

using airbc::Attr;
using airbc::AttrList;
using airbc::Instruction;
using airbc::Metadata;

namespace {
const char *kDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-"
    "v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-"
    "v1024:1024:1024-n8:16:32";

std::vector<Attr> attrs(std::initializer_list<const char *> names) {
  std::vector<Attr> r;
  for (const char *n : names) r.push_back(Attr::get(n));
  return r;
}

// Declaration / call-site attribute sets per intrinsic family, as metal emits them (catalogue.md).
void familyAttrs(Kernel::Attrs a, std::vector<Attr> &decl, std::vector<Attr> &site) {
  switch (a) {
  case Kernel::Attrs::Pure:
    decl = attrs({"mustprogress", "nofree", "nosync", "nounwind", "readnone", "willreturn"});
    site = attrs({"nounwind", "readnone", "willreturn"});
    break;
  case Kernel::Attrs::Convergent:
    decl = attrs({"convergent", "mustprogress", "nounwind", "willreturn"});
    site = attrs({"convergent", "nounwind", "willreturn"});
    break;
  case Kernel::Attrs::ConvergentRead:
    decl = attrs({"convergent", "mustprogress", "nofree", "nounwind", "readonly", "willreturn"});
    site = attrs({"convergent", "nounwind", "readonly", "willreturn"});
    break;
  case Kernel::Attrs::ConvergentWrite:
    decl = attrs({"convergent", "mustprogress", "nounwind", "willreturn", "writeonly"});
    site = attrs({"convergent", "nounwind", "willreturn", "writeonly"});
    break;
  case Kernel::Attrs::TexSample:
    decl = attrs({"argmemonly", "convergent", "mustprogress", "nofree", "nounwind", "readonly", "willreturn"});
    site = attrs({"argmemonly", "convergent", "nounwind", "readonly", "willreturn"});
    break;
  case Kernel::Attrs::TexRead:
    decl = attrs({"argmemonly", "mustprogress", "nofree", "nounwind", "readonly", "willreturn"});
    site = attrs({"argmemonly", "nounwind", "readonly", "willreturn"});
    break;
  case Kernel::Attrs::TexWrite:
    decl = attrs({"argmemonly", "mustprogress", "nounwind", "willreturn"});
    site = attrs({"argmemonly", "nounwind", "willreturn"});
    break;
  case Kernel::Attrs::Atomic:
    decl = attrs({"mustprogress", "nounwind", "willreturn"});
    site = attrs({"nounwind", "willreturn"});
    break;
  }
}

bool isFP(const Type *t) {
  if (t->kind == Type::Vector) t = t->elem;
  return t->kind == Type::Half || t->kind == Type::Float || t->kind == Type::Double;
}
const Type *scalarOf(const Type *t) { return t->kind == Type::Vector ? t->elem : t; }

const char *builtinName(Builtin b) {
  switch (b) {
  case Builtin::ThreadPositionInGrid: return "air.thread_position_in_grid";
  case Builtin::ThreadPositionInThreadgroup: return "air.thread_position_in_threadgroup";
  case Builtin::ThreadgroupPositionInGrid: return "air.threadgroup_position_in_grid";
  case Builtin::ThreadsPerThreadgroup: return "air.threads_per_threadgroup";
  case Builtin::ThreadgroupsPerGrid: return "air.threadgroups_per_grid";
  case Builtin::ThreadsPerGrid: return "air.threads_per_grid";
  case Builtin::ThreadIndexInThreadgroup: return "air.thread_index_in_threadgroup";
  case Builtin::ThreadIndexInSimdgroup: return "air.thread_index_in_simdgroup";
  case Builtin::SimdgroupIndexInThreadgroup: return "air.simdgroup_index_in_threadgroup";
  case Builtin::ThreadsPerSimdgroup: return "air.threads_per_simdgroup";
  case Builtin::SimdgroupsPerThreadgroup: return "air.simdgroups_per_threadgroup";
  }
  return "";
}

const char *accessMD(Access a) {
  switch (a) {
  case Access::Read: return "air.read";
  case Access::ReadWrite: return "air.read_write";
  case Access::Write: return "air.write";
  case Access::Sample: return "air.sample";
  }
  return "";
}
const char *accessName(Access a) {
  switch (a) {
  case Access::Read: return "read";
  case Access::ReadWrite: return "read_write";
  case Access::Write: return "write";
  case Access::Sample: return "sample";
  }
  return "";
}
// Last immediate of read/write_texture: the texture's declared access.
const char* texelName(Texel t) {
  switch (t) {
    case Texel::Float:
      return "float";
    case Texel::Half:
      return "half";
    case Texel::Int:
      return "int";
    case Texel::Uint:
      return "uint";
  }
  return "float";
}

// Intrinsic suffix: v4f32, v4f16, s.v4i32, u.v4i32.
const char* texelSuffix(Texel t) {
  switch (t) {
    case Texel::Float:
      return "v4f32";
    case Texel::Half:
      return "v4f16";
    case Texel::Int:
      return "s.v4i32";
    case Texel::Uint:
      return "u.v4i32";
  }
  return "v4f32";
}

uint32_t accessCode(Access a) {
  switch (a) {
  case Access::Sample: return 0;
  case Access::Read: return 1;
  case Access::Write: return 2;
  case Access::ReadWrite: return 3;
  }
  return 0;
}

uint32_t scalarBytes(const Type *t) {
  switch (t->kind) {
  case Type::Half: return 2;
  case Type::Float: return 4;
  case Type::Double: return 8;
  case Type::Int: return t->n <= 8 ? 1 : t->n / 8;
  case Type::Pointer: return 8;
  default: return 0;
  }
}
}  // namespace

Kernel::Kernel(std::string name, Options opts) : name_(std::move(name)), opts_(std::move(opts)) {
  m.triple = "air64_v26-apple-macosx14.0.0";
  m.datalayout = kDataLayout;
  m.sourceFileName = opts_.sourceName;
  f16 = m.halfTy();
  f32 = m.floatTy();
  i1 = m.intTy(1);
  i8 = m.intTy(8);
  i16 = m.intTy(16);
  i32 = m.intTy(32);
  i64 = m.intTy(64);
  fmf = opts_.fastMath ? airbc::FMF_Fast : airbc::FMF_Strict;
}

std::string Kernel::typeName(Type *t) const {
  if (t->kind == Type::Vector) return typeName(t->elem) + std::to_string(t->n);
  switch (t->kind) {
  case Type::Half: return "half";
  case Type::Float: return "float";
  case Type::Double: return "double";
  case Type::Int:
    switch (t->n) {
    case 1: return "bool";
    case 8: return "uchar";
    case 16: return "ushort";
    case 32: return "uint";
    case 64: return "ulong";
    }
    break;
  default: break;
  }
  throw std::runtime_error("airkit: give typeName/size/align explicitly for this buffer element type");
}

std::string Kernel::mangle(Type *t, bool isSigned, bool intSign) const {
  std::string pre = intSign && scalarOf(t)->kind == Type::Int ? (isSigned ? "s." : "u.") : "";
  auto scalar = [](const Type *s) -> std::string {
    if (s->kind == Type::Half) return "f16";
    if (s->kind == Type::Float) return "f32";
    if (s->kind == Type::Double) return "f64";
    return "i" + std::to_string(s->n);
  };
  if (t->kind == Type::Vector) return pre + "v" + std::to_string(t->n) + scalar(t->elem);
  return pre + scalar(t);
}

// ---------------------------------------------------------------- arguments
unsigned Kernel::buffer(Type *elem, unsigned location, Access access, const std::string &name, unsigned addrSpace,
                        std::string tn, uint32_t size, uint32_t align) {
  if (fn_) throw std::runtime_error("airkit: arguments must be declared before begin()");
  if (tn.empty()) tn = typeName(elem);
  if (!size) {
    uint32_t s = scalarBytes(scalarOf(elem));
    uint32_t n = elem->kind == Type::Vector ? (elem->n == 3 ? 4 : elem->n) : 1;
    size = s * n;
  }
  if (!align) align = size;
  args_.push_back({m.ptrTy(elem, addrSpace),
                   {S("air.buffer"), S("air.location_index"), I(location), I(1), S(accessMD(access)),
                    S("air.address_space"), I(addrSpace), S("air.arg_type_size"), I(size),
                    S("air.arg_type_align_size"), I(align), S("air.arg_type_name"), S(tn), S("air.arg_name"),
                    S(name)}});
  return unsigned(args_.size() - 1);
}

unsigned Kernel::threadgroupBuffer(Type *elem, unsigned location, const std::string &name, std::string tn,
                                   uint32_t size, uint32_t align) {
  return buffer(elem, location, Access::ReadWrite, name, 3, std::move(tn), size, align);
}

unsigned Kernel::builtin(Builtin b, unsigned width, bool ushort, const std::string &name) {
  if (fn_) throw std::runtime_error("airkit: arguments must be declared before begin()");
  Type *s = ushort ? i16 : i32;
  Type *t = width > 1 ? m.vecTy(s, width) : s;
  std::string tn = std::string(ushort ? "ushort" : "uint") + (width > 1 ? std::to_string(width) : "");
  std::string an = name.empty() ? std::string(builtinName(b)).substr(4) : name;
  args_.push_back({t, {S(builtinName(b)), S("air.arg_type_name"), S(tn), S("air.arg_name"), S(an)}});
  return unsigned(args_.size() - 1);
}

unsigned Kernel::texture2d(unsigned location, Access access, Texel texel, const std::string& name) {
  if (fn_) throw std::runtime_error("airkit: arguments must be declared before begin()");
  if (!texTy_) texTy_ = m.structTy("struct._texture_2d_t");
  std::string tn = std::string("texture2d<") + texelName(texel) + ", " + accessName(access) + ">";
  args_.push_back({m.ptrTy(texTy_, 1),
                   {S("air.texture"), S("air.location_index"), I(location), I(1), S(accessMD(access)),
                    S("air.arg_type_name"), S(tn), S("air.arg_name"), S(name)}});
  return unsigned(args_.size() - 1);
}

unsigned Kernel::sampler(unsigned location, const std::string &name) {
  if (fn_) throw std::runtime_error("airkit: arguments must be declared before begin()");
  if (!smpTy_) smpTy_ = m.structTy("struct._sampler_t");
  args_.push_back({m.ptrTy(smpTy_, 2),
                   {S("air.sampler"), S("air.location_index"), I(location), I(1), S("air.arg_type_name"),
                    S("sampler"), S("air.arg_name"), S(name)}});
  return unsigned(args_.size() - 1);
}

Builder &Kernel::begin() {
  if (fn_) return *b_;
  std::vector<Type *> params;
  for (const Arg &a : args_) params.push_back(a.type);
  fn_ = m.addFunction(name_, m.fnTy(m.voidTy(), params));
  fn_->attrs = m.attrList({Attr::get("nounwind"), Attr::str("frame-pointer", "all"), Attr::str("no-builtins")});
  b_ = std::make_unique<Builder>(m, m.addBlock(fn_));
  return *b_;
}

Value *Kernel::arg(unsigned i) const {
  if (!fn_) throw std::runtime_error("airkit: call begin() first");
  return fn_->args.at(i);
}

airbc::BasicBlock *Kernel::block(const std::string &name) { return m.addBlock(fn_, name); }

// ---------------------------------------------------------------- intrinsic plumbing
Function *Kernel::declare(const std::string &name, Type *ret, const std::vector<Type *> &params, Attrs a,
                          const std::vector<std::vector<Attr>> &paramAttrs) {
  Function *&f = decls_[name];
  if (f) return f;
  std::vector<Attr> decl, site;
  familyAttrs(a, decl, site);
  f = m.addFunction(name, m.fnTy(ret, params));
  f->attrs = m.attrList(decl, {}, paramAttrs);
  f->unnamedAddr = 2;  // local_unnamed_addr
  return f;
}

Value *Kernel::callIntrinsic(const std::string &name, Type *ret, const std::vector<Value *> &args, Attrs a,
                             const std::vector<std::vector<Attr>> &paramAttrs, bool fpFlags) {
  if (!fn_) throw std::runtime_error("airkit: call begin() first");
  std::vector<Type *> params;
  for (Value *v : args) params.push_back(v->type);
  Function *f = declare(name, ret, params, a, paramAttrs);
  std::vector<Attr> decl, site;
  familyAttrs(a, decl, site);
  Instruction *c = b_->call(f, args, m.attrList(site), "", fpFlags ? fmf : 0);
  c->tail = true;
  return c;
}

Value *Kernel::toI16(Value *v) {
  if (v->type == i16) return v;
  if (v->vk == Value::Const) return m.constInt(i16, int64_t(static_cast<Constant *>(v)->bits));
  return b_->cast(airbc::Trunc, v, i16);
}

// ---------------------------------------------------------------- sync / simd
void Kernel::barrier(uint32_t mem) {
  callIntrinsic("air.wg.barrier", m.voidTy(), {m.constInt(i32, mem), m.constInt(i32, 1)}, Attrs::Convergent);
}
void Kernel::simdgroupBarrier(uint32_t mem) {
  callIntrinsic("air.simdgroup.barrier", m.voidTy(), {m.constInt(i32, mem), m.constInt(i32, 1)},
                Attrs::Convergent);
}

Value *Kernel::simdShuffle(Shuffle op, Value *v, Value *lane, bool isSigned) {
  static const char *names[] = {"air.simd_shuffle", "air.simd_shuffle_xor", "air.simd_shuffle_up",
                                "air.simd_shuffle_down", "air.simd_broadcast"};
  return callIntrinsic(std::string(names[int(op)]) + "." + mangle(v->type, isSigned, true), v->type,
                       {v, toI16(lane)}, Attrs::Convergent, {}, isFP(v->type));
}
Value *Kernel::simdBallot(Value *pred) {
  return callIntrinsic("air.simd_ballot.i64", i64, {pred}, Attrs::Convergent);
}
Value *Kernel::simdBroadcastFirst(Value *v, bool isSigned) {
  return callIntrinsic("air.simd_broadcast_first." + mangle(v->type, isSigned, true), v->type, {v},
                       Attrs::Convergent, {}, isFP(v->type));
}
Value *Kernel::simdSum(Value *v, bool isSigned) {
  return callIntrinsic("air.simd_sum." + mangle(v->type, isSigned, true), v->type, {v}, Attrs::Convergent, {},
                       isFP(v->type));
}

// ---------------------------------------------------------------- math / conversion
Value *Kernel::math(const std::string &op, Value *x, bool fastVariant) {
  return callIntrinsic("air." + std::string(fastVariant ? "fast_" : "") + op + "." + mangle(x->type, false, false),
                       x->type, {x}, Attrs::Pure, {}, isFP(x->type));
}
Value *Kernel::math2(const std::string &op, Value *x, Value *y, bool fastVariant) {
  return callIntrinsic("air." + std::string(fastVariant ? "fast_" : "") + op + "." + mangle(x->type, false, false),
                       x->type, {x, y}, Attrs::Pure, {}, isFP(x->type));
}
Value *Kernel::fma(Value *a, Value *b, Value *c) {
  return callIntrinsic("air.fma." + mangle(a->type, false, false), a->type, {a, b, c}, Attrs::Pure, {}, true);
}

Value *Kernel::convert(Value *v, Type *to, bool srcSigned, bool dstSigned) {
  Type *from = v->type;
  const Type *fs = scalarOf(from), *ts = scalarOf(to);
  bool ff = isFP(from), tf = isFP(to);
  if (ff && tf) {
    if (scalarBytes(fs) == scalarBytes(ts)) return v;
    return b_->cast(scalarBytes(ts) < scalarBytes(fs) ? airbc::FPTrunc : airbc::FPExt, v, to);
  }
  if (!ff && !tf) {
    if (ts->n == fs->n) return v;
    if (ts->n < fs->n) return b_->cast(airbc::Trunc, v, to);
    return b_->cast(srcSigned ? airbc::SExt : airbc::ZExt, v, to);
  }
  // air.convert.<dst kind>.<dst type>.<src kind>.<src type>, kind f / s / u.
  std::string name = "air.convert." + std::string(tf ? "f." : dstSigned ? "s." : "u.") + mangle(to, false, false) +
                     "." + (ff ? "f." : srcSigned ? "s." : "u.") + mangle(from, false, false);
  return callIntrinsic(name, to, {v}, Attrs::Pure, {}, tf);
}

// ---------------------------------------------------------------- textures
Constant *Kernel::constSampler(uint64_t state) {
  if (!smpTy_) smpTy_ = m.structTy("struct._sampler_t");
  std::string gname = "__air_sampler_state";
  if (!samplers_.empty()) gname += "." + std::to_string(samplers_.size());
  airbc::GlobalVariable *g =
      m.addGlobal(gname, i64, 2, m.constInt(i64, int64_t(state)), 8, airbc::Link_Internal, 0, true);
  samplers_.push_back(g);
  return m.constCast(airbc::BitCast, g, m.ptrTy(smpTy_, 2));
}

Value *Kernel::sample2d(Value *tex, Value *smp, Value *uv, bool halfTexels, Value *lod) {
  Type *texel = m.vecTy(halfTexels ? f16 : f32, 4);
  Type *ret = m.literalStructTy({texel, i8});
  std::vector<Value *> a = {tex,
                            smp,
                            uv,
                            m.constInt(i1, 1),
                            m.constNull(m.vecTy(i32, 2)),
                            m.constInt(i1, lod ? 1 : 0),
                            lod ? lod : m.constFloat(0.0f),
                            m.constFloat(0.0f),
                            m.constInt(i32, 0)};
  std::vector<Attr> ro = attrs({"nocapture", "readonly"});
  Value *r = callIntrinsic(std::string("air.sample_texture_2d.") + (halfTexels ? "v4f16" : "v4f32"), ret, a,
                           Attrs::TexSample, {ro, ro});
  return b_->extractValue(r, {0});
}

Value* Kernel::read2d(Value* tex, Value* coord, Access access, Texel kind, Value* lod) {
  Type* texel = m.vecTy(kind == Texel::Half ? f16 : kind == Texel::Float ? f32 : i32, 4);
  Value* r = callIntrinsic(std::string("air.read_texture_2d.") + texelSuffix(kind), m.literalStructTy({texel, i8}),
                           {tex, coord, lod ? lod : m.constInt(i32, 0), m.constInt(i32, accessCode(access))},
                           Attrs::TexRead, {attrs({"nocapture", "readonly"})});
  return b_->extractValue(r, {0});
}

void Kernel::write2d(Value *tex, Value *coord, Value *texel, Access access, Value *lod) {
  Type::Kind k = scalarOf(texel->type)->kind;
  if (k != Type::Half && k != Type::Float)
    throw std::runtime_error("airkit: integer texel writes need write2d(..., Texel::Int or Texel::Uint, ...)");
  write2d(tex, coord, texel, access, k == Type::Half ? Texel::Half : Texel::Float, lod);
}

void Kernel::write2d(Value* tex, Value* coord, Value* texel, Access access, Texel kind, Value* lod) {
  callIntrinsic(std::string("air.write_texture_2d.") + texelSuffix(kind), m.voidTy(),
                {tex, coord, texel, lod ? lod : m.constInt(i32, 0), m.constInt(i32, accessCode(access))},
                Attrs::TexWrite, {attrs({"nocapture"})});
}

Value *Kernel::textureWidth(Value *tex, Value *lod) {
  return callIntrinsic("air.get_width_texture_2d", i32, {tex, lod ? lod : m.constInt(i32, 0)}, Attrs::TexRead,
                       {attrs({"nocapture", "readonly"})});
}
Value *Kernel::textureHeight(Value *tex, Value *lod) {
  return callIntrinsic("air.get_height_texture_2d", i32, {tex, lod ? lod : m.constInt(i32, 0)}, Attrs::TexRead,
                       {attrs({"nocapture", "readonly"})});
}

// ---------------------------------------------------------------- simdgroup_matrix 8x8
Value *Kernel::matFill(Type *elem, Value *scalar) {
  Type *t = m.vecTy(elem, 64);
  return callIntrinsic("air.simdgroup_matrix_8x8_init_filled." + mangle(t, false, false) + "." +
                           mangle(elem, false, false),
                       t, {scalar}, Attrs::Convergent, {}, true);
}

Value *Kernel::matLoad(Type *elem, Value *ptr, Value *stride, bool transpose, Value *origin) {
  Type *t = m.vecTy(elem, 64);
  std::string p = "p" + std::to_string(ptr->type->n) + mangle(elem, false, false);
  return callIntrinsic("air.simdgroup_matrix_8x8_load." + mangle(t, false, false) + "." + p, t,
                       {ptr, stride, origin ? origin : m.constNull(m.vecTy(i64, 2)), m.constInt(i1, transpose)},
                       Attrs::ConvergentRead, {attrs({"nocapture", "readonly"})}, true);
}

void Kernel::matStore(Value *mat, Value *ptr, Value *stride, bool transpose, Value *origin) {
  std::string p = "p" + std::to_string(ptr->type->n) + mangle(mat->type->elem, false, false);
  callIntrinsic("air.simdgroup_matrix_8x8_store." + mangle(mat->type, false, false) + "." + p, m.voidTy(),
                {mat, ptr, stride, origin ? origin : m.constNull(m.vecTy(i64, 2)), m.constInt(i1, transpose)},
                Attrs::ConvergentWrite, {{}, attrs({"nocapture", "writeonly"})});
}

Value *Kernel::matMulAcc(Value *a, Value *b, Value *c) {
  return callIntrinsic("air.simdgroup_matrix_8x8_multiply_accumulate." + mangle(c->type, false, false) + "." +
                           mangle(a->type, false, false) + "." + mangle(b->type, false, false) + "." +
                           mangle(c->type, false, false),
                       c->type, {a, b, c}, Attrs::Convergent, {}, true);
}

// ---------------------------------------------------------------- atomics
Value *Kernel::atomicRMW(AtomicOp op, Value *ptr, Value *val, bool isSigned) {
  static const char *names[] = {"add", "sub", "min", "max", "and", "or", "xor", "xchg"};
  bool local = ptr->type->n == 3;
  std::string ty = op == AtomicOp::Xchg ? mangle(val->type, false, false) : mangle(val->type, isSigned, true);
  std::string name = std::string("air.atomic.") + (local ? "local." : "global.") + names[int(op)] + "." + ty;
  return callIntrinsic(name, val->type,
                       {ptr, val, m.constInt(i32, 0), m.constInt(i32, local ? 1 : 2), m.constInt(i1, 1)},
                       Attrs::Atomic, {attrs({"nocapture"})}, isFP(val->type));
}

Value *Kernel::atomicLoad(Value *ptr) {
  bool local = ptr->type->n == 3;
  Type *t = ptr->type->elem;
  return callIntrinsic(std::string("air.atomic.") + (local ? "local" : "global") + ".load." + mangle(t, false, false),
                       t, {ptr, m.constInt(i32, 0), m.constInt(i32, local ? 1 : 2), m.constInt(i1, 1)},
                       Attrs::Atomic, {attrs({"nocapture"})});
}

void Kernel::atomicStore(Value *ptr, Value *val) {
  bool local = ptr->type->n == 3;
  callIntrinsic(std::string("air.atomic.") + (local ? "local" : "global") + ".store." +
                    mangle(val->type, false, false),
                m.voidTy(), {ptr, val, m.constInt(i32, 0), m.constInt(i32, local ? 1 : 2), m.constInt(i1, 1)},
                Attrs::Atomic, {attrs({"nocapture"})});
}

Value *Kernel::atomicCmpxchgWeak(Value *ptr, Value *expected, Value *desired) {
  bool local = ptr->type->n == 3;
  return callIntrinsic(std::string("air.atomic.") + (local ? "local" : "global") + ".cmpxchg.weak." +
                           mangle(desired->type, false, false),
                       desired->type,
                       {ptr, expected, desired, m.constInt(i32, 0), m.constInt(i32, 0),
                        m.constInt(i32, local ? 1 : 2), m.constInt(i1, 1)},
                       Attrs::Atomic, {attrs({"nocapture"}), attrs({"nocapture"})});
}

// ---------------------------------------------------------------- output
void Kernel::finalize() {
  if (finalized_) return;
  finalized_ = true;
  begin();
  auto N = [&](std::vector<Metadata *> ops) { return m.mdNode(std::move(ops)); };
  auto flag = [&](int b, const char *k, int v) { return N({I(b), S(k), I(v)}); };
  m.addNamedMD("llvm.module.flags",
               {flag(1, "wchar_size", 4), flag(7, "air.max_device_buffers", 31),
                flag(7, "air.max_constant_buffers", 31), flag(7, "air.max_threadgroup_buffers", 31),
                flag(7, "air.max_textures", 128), flag(7, "air.max_read_write_textures", 8),
                flag(7, "air.max_samplers", 16)});
  std::vector<Metadata *> argMD;
  for (unsigned i = 0; i < args_.size(); ++i) {
    std::vector<Metadata *> ops = {I(i)};
    ops.insert(ops.end(), args_[i].md.begin(), args_[i].md.end());
    argMD.push_back(N(ops));
  }
  std::vector<Metadata *> kernel = {m.mdValue(fn_), N({}), N(argMD)};
  if (opts_.maxWorkGroupSize) kernel.push_back(N({S("air.max_work_group_size"), I(opts_.maxWorkGroupSize)}));
  m.addNamedMD("air.kernel", {N(kernel)});
  std::vector<Metadata *> co;
  if (opts_.denormsDisable) co.push_back(N({S("air.compile.denorms_disable")}));
  co.push_back(N({S(opts_.fastMath ? "air.compile.fast_math_enable" : "air.compile.fast_math_disable")}));
  co.push_back(N({S("air.compile.framebuffer_fetch_enable")}));
  m.addNamedMD("air.compile_options", co);
  if (!samplers_.empty()) {
    std::vector<Metadata *> ss;
    for (airbc::GlobalVariable *g : samplers_) ss.push_back(N({S("air.sampler_state"), m.mdValue(g)}));
    m.addNamedMD("air.sampler_states", ss);
  }
  m.addNamedMD("llvm.ident", {N({S("airkit")})});
  m.addNamedMD("air.version", {N({I(2), I(6), I(0)})});
  m.addNamedMD("air.language_version", {N({S("Metal"), I(3), I(1), I(0)})});
  m.addNamedMD("air.source_file_name", {N({S(opts_.sourceName)})});
}

std::vector<uint8_t> Kernel::bitcode() {
  finalize();
  return airbc::writeBitcode(m);
}

std::vector<uint8_t> buildMetallib(const std::vector<LibraryEntry> &kernels) {
  std::vector<mtlb::Function> fns(kernels.size());
  for (size_t i = 0; i < kernels.size(); ++i) {
    fns[i].name = kernels[i].name;
    fns[i].type = mtlb::FunctionType::Kernel;
    fns[i].bitcode = kernels[i].bitcode.data();
    fns[i].bitcode_size = kernels[i].bitcode.size();
  }
  std::vector<uint8_t> out(mtlb::library_size(fns.data(), fns.size()));
  out.resize(mtlb::write(out.data(), mtlb::Target{}, fns.data(), fns.size()));
  return out;
}

}  // namespace airkit
