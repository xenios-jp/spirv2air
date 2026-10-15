// airbc: LLVM-13-layout typed-pointer bitcode writer for AIR. See airbc.h.
//
// Structure follows LLVM 13 (lib/Bitcode/Writer/{BitcodeWriter,ValueEnumerator}.cpp, Bitstream/BitstreamWriter.h,
// Object/IRSymtab.cpp); comments name the LLVM function each piece corresponds to.
#include "airbc.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace airbc {

// ================================================================ attributes
namespace {
struct AttrInfo {
  const char *ir, *llvm; // IR spelling, LLVM 13 AttrKind enumerator name (enum order = strcmp order per class)
  uint32_t code;         // bitc::ATTR_KIND_*
  uint8_t cls;           // 0 enum, 1 type, 2 int  (LLVM orders enum < type < int)
};
const AttrInfo kAttrs[] = {
    {"alwaysinline", "AlwaysInline", 2, 0},     {"argmemonly", "ArgMemOnly", 45, 0},
    {"builtin", "Builtin", 35, 0},              {"cold", "Cold", 36, 0},
    {"convergent", "Convergent", 43, 0},        {"hot", "Hot", 72, 0},
    {"immarg", "ImmArg", 60, 0},                {"inreg", "InReg", 5, 0},
    {"inaccessiblememonly", "InaccessibleMemOnly", 49, 0},
    {"inaccessiblemem_or_argmemonly", "InaccessibleMemOrArgMemOnly", 50, 0},
    {"inlinehint", "InlineHint", 4, 0},         {"mustprogress", "MustProgress", 70, 0},
    {"noalias", "NoAlias", 9, 0},               {"nobuiltin", "NoBuiltin", 10, 0},
    {"nocallback", "NoCallback", 71, 0},        {"nocapture", "NoCapture", 11, 0},
    {"noduplicate", "NoDuplicate", 12, 0},      {"nofree", "NoFree", 62, 0},
    {"noinline", "NoInline", 14, 0},            {"nomerge", "NoMerge", 66, 0},
    {"norecurse", "NoRecurse", 48, 0},          {"noreturn", "NoReturn", 17, 0},
    {"nosync", "NoSync", 63, 0},                {"noundef", "NoUndef", 68, 0},
    {"nounwind", "NoUnwind", 18, 0},            {"nonnull", "NonNull", 39, 0},
    {"optnone", "OptimizeNone", 37, 0},         {"readnone", "ReadNone", 20, 0},
    {"readonly", "ReadOnly", 21, 0},            {"returned", "Returned", 22, 0},
    {"signext", "SExt", 24, 0},                 {"speculatable", "Speculatable", 53, 0},
    {"willreturn", "WillReturn", 61, 0},        {"writeonly", "WriteOnly", 52, 0},
    {"zeroext", "ZExt", 34, 0},
    {"align", "Alignment", 1, 2},               {"dereferenceable", "Dereferenceable", 41, 2},
    {"dereferenceable_or_null", "DereferenceableOrNull", 42, 2},
};
int attrRank(const AttrInfo &a) {
  int r = 0;
  for (const AttrInfo &o : kAttrs)
    if (o.cls < a.cls || (o.cls == a.cls && std::strcmp(o.llvm, a.llvm) < 0)) ++r;
  return r;
}
} // namespace

Attr Attr::get(const char *name, uint64_t intValue) {
  for (const AttrInfo &a : kAttrs)
    if (!std::strcmp(a.ir, name)) {
      Attr r;
      r.cls = a.cls == 2 ? Int : Enum;
      r.code = a.code;
      r.rank = attrRank(a);
      r.value = intValue;
      return r;
    }
  throw std::runtime_error(std::string("airbc: unknown attribute ") + name);
}

Attr Attr::str(std::string k, std::string v) {
  Attr r;
  r.cls = v.empty() ? Str : StrVal;
  r.key = std::move(k);
  r.val = std::move(v);
  return r;
}

// AttributeImpl::operator<: enum/int kinds numerically, then string attributes lexicographically.
bool Attr::operator<(const Attr &o) const {
  bool s = cls >= Str, os = o.cls >= Str;
  if (s != os) return !s;
  if (!s) return rank != o.rank ? rank < o.rank : value < o.value;
  return key != o.key ? key < o.key : val < o.val;
}

// ================================================================ module
Module::Module() {
  void_ = newType(Type::Void);
  half_ = newType(Type::Half);
  float_ = newType(Type::Float);
  double_ = newType(Type::Double);
  label_ = newType(Type::Label);
  metadata_ = newType(Type::Metadata);
}
Module::~Module() = default;

Type *Module::newType(Type::Kind k) {
  types_.push_back(std::make_unique<Type>());
  types_.back()->kind = k;
  return types_.back().get();
}

// Intern keys: a tag, then fixed-width raw fields, so no two keys of one map can collide. Short keys stay in
// std::string's inline buffer.
static std::string ptrKey(const void *p) { return std::string(reinterpret_cast<const char *>(&p), sizeof p); }
static std::string numKey(uint64_t v) { return std::string(reinterpret_cast<const char *>(&v), sizeof v); }

Type *Module::intTy(unsigned w) {
  if (w < 65 && intTys_[w]) return intTys_[w];
  Type *&t = typeIntern_["i" + numKey(w)];
  if (!t) (t = newType(Type::Int))->n = w;
  if (w < 65) intTys_[w] = t;
  return t;
}
Type *Module::ptrTy(Type *pointee, unsigned as) {
  Type *&t = typeIntern_["p" + ptrKey(pointee) + numKey(as)];
  if (!t) {
    t = newType(Type::Pointer);
    t->elem = pointee;
    t->n = as;
  }
  return t;
}
Type *Module::vecTy(Type *elem, unsigned n) {
  Type *&t = typeIntern_["v" + ptrKey(elem) + numKey(n)];
  if (!t) {
    t = newType(Type::Vector);
    t->elem = elem;
    t->n = n;
  }
  return t;
}
Type *Module::arrayTy(Type *elem, unsigned n) {
  Type *&t = typeIntern_["a" + ptrKey(elem) + numKey(n)];
  if (!t) {
    t = newType(Type::Array);
    t->elem = elem;
    t->n = n;
  }
  return t;
}
Type *Module::fnTy(Type *ret, std::vector<Type *> params, bool vararg) {
  std::string k = "f" + ptrKey(ret) + numKey(vararg);
  for (Type *p : params) k += ptrKey(p);
  Type *&t = typeIntern_[k];
  if (!t) {
    t = newType(Type::Function);
    t->elem = ret;
    t->elems = std::move(params);
    t->vararg = vararg;
  }
  return t;
}
Type *Module::literalStructTy(std::vector<Type *> elems, bool packed) {
  std::string k = "s" + numKey(packed);
  for (Type *e : elems) k += ptrKey(e);
  Type *&t = typeIntern_[k];
  if (!t) {
    t = newType(Type::Struct);
    t->elems = std::move(elems);
    t->packed = packed;
    t->literal = true;
  }
  return t;
}
Type *Module::structTy(const std::string &name) {
  Type *t = newType(Type::Struct);
  t->name = name;
  t->opaque = true;
  return t;
}
void Module::setBody(Type *st, std::vector<Type *> elems, bool packed) {
  st->elems = std::move(elems);
  st->packed = packed;
  st->opaque = false;
}

static uint64_t truncBits(uint64_t v, unsigned w) { return w >= 64 ? v : v & ((1ull << w) - 1); }

Constant *Module::constInt(Type *t, int64_t v) {
  assert(t->kind == Type::Int);
  uint64_t bits = truncBits((uint64_t)v, t->n);
  if (!bits) return constNull(t);
  ConstSlot &slot = constCache_[(uintptr_t(t) >> 4 ^ bits * 0x9E3779B97F4A7C15ull >> 40) & 255];
  if (slot.c && slot.t == t && slot.bits == bits && !slot.fp) return slot.c;
  Constant *&c = constIntern_["i" + ptrKey(t) + numKey(bits)];
  if (!c) (c = make<Constant>(t, Constant::Int))->bits = bits;
  slot = {t, bits, c, false};
  return c;
}
Constant *Module::constFPBits(Type *t, uint64_t bits) {
  if (!bits) return constNull(t);
  ConstSlot &slot = constCache_[(uintptr_t(t) >> 4 ^ bits * 0x9E3779B97F4A7C15ull >> 40) & 255];
  if (slot.c && slot.t == t && slot.bits == bits && slot.fp) return slot.c;
  Constant *&c = constIntern_["f" + ptrKey(t) + numKey(bits)];
  if (!c) (c = make<Constant>(t, Constant::FP))->bits = bits;
  slot = {t, bits, c, true};
  return c;
}
Constant *Module::constFloat(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return constFPBits(float_, b);
}
Constant *Module::constDouble(double d) {
  uint64_t b;
  std::memcpy(&b, &d, 8);
  return constFPBits(double_, b);
}
Constant *Module::constNull(Type *t) {
  if (t->nullC) return t->nullC;
  Constant *&c = constIntern_["n" + ptrKey(t)];
  if (!c) c = make<Constant>(t, Constant::Null);
  return t->nullC = c;
}
Constant *Module::constUndef(Type *t) {
  if (t->undefC) return t->undefC;
  Constant *&c = constIntern_["u" + ptrKey(t)];
  if (!c) c = make<Constant>(t, Constant::Undef);
  return t->undefC = c;
}
Constant *Module::constPoison(Type *t) {
  if (t->poisonC) return t->poisonC;
  Constant *&c = constIntern_["P" + ptrKey(t)];
  if (!c) c = make<Constant>(t, Constant::Poison);
  return t->poisonC = c;
}
Constant *Module::constAggregate(Type *t, std::vector<Constant *> ops) {
  bool allNull = true;
  for (Constant *o : ops) allNull &= o->ck == Constant::Null;
  if (allNull) return constNull(t); // ConstantAggregateZero
  std::string k = "A" + ptrKey(t);
  for (Constant *o : ops) k += ptrKey(o);
  Constant *&c = constIntern_[k];
  if (!c) {
    c = make<Constant>(t, Constant::Aggregate);
    c->ops = std::move(ops);
  }
  return c;
}

static Type *indexedType(Type *t, const std::vector<Value *> &idx, size_t from);

Constant *Module::constCast(CastOpc opc, Constant *v, Type *to) {
  Constant *&c = constIntern_["Ec" + numKey(opc) + ptrKey(v) + ptrKey(to)];
  if (!c) {
    c = make<Constant>(to, Constant::Expr);
    c->eop = Constant::CE_Cast;
    c->sub = opc;
    c->ops = {v};
  }
  return c;
}
Constant *Module::constGEP(Type *src, Constant *ptr, std::vector<Constant *> idx, bool inbounds) {
  std::string k = "Eg" + numKey(inbounds) + ptrKey(src) + ptrKey(ptr);
  for (Constant *i : idx) k += ptrKey(i);
  Constant *&c = constIntern_[k];
  if (!c) {
    std::vector<Value *> iv(idx.begin(), idx.end());
    c = make<Constant>(ptrTy(indexedType(src, iv, 1), ptr->type->n), Constant::Expr);
    c->eop = Constant::CE_GEP;
    c->srcElemTy = src;
    c->inbounds = inbounds;
    c->ops = {ptr};
    c->ops.insert(c->ops.end(), idx.begin(), idx.end());
  }
  return c;
}
Constant *Module::constBinOp(BinOpc opc, Constant *a, Constant *b, uint8_t flags) {
  Constant *&c = constIntern_["Eb" + numKey(opc) + numKey(flags) + ptrKey(a) + ptrKey(b)];
  if (!c) {
    c = make<Constant>(a->type, Constant::Expr);
    c->eop = Constant::CE_BinOp;
    c->sub = opc;
    c->flags = flags;
    c->ops = {a, b};
  }
  return c;
}

GlobalVariable *Module::addGlobal(const std::string &name, Type *valueType, unsigned addrSpace, Constant *init,
                                  uint32_t align, uint32_t linkage, uint32_t unnamedAddr, bool isConstant) {
  GlobalVariable *g = make<GlobalVariable>(ptrTy(valueType, addrSpace), valueType);
  g->name = name;
  g->init = init;
  g->align = align;
  g->linkage = linkage;
  g->unnamedAddr = unnamedAddr;
  g->isConstant = isConstant;
  globals.push_back(g);
  return g;
}

Function *Module::addFunction(const std::string &name, Type *fty) {
  Function *f = make<Function>(ptrTy(fty, 0), fty);
  f->name = name;
  for (unsigned i = 0; i < fty->elems.size(); ++i) f->args.push_back(make<Argument>(fty->elems[i], i));
  functions.push_back(f);
  return f;
}
BasicBlock *Module::addBlock(Function *f, const std::string &name) {
  BasicBlock *bb = make<BasicBlock>(label_);
  bb->name = name;
  f->blocks.push_back(bb);
  return bb;
}
Instruction *Module::newInst(Op op, Type *t) {
  insts_.emplace_back(op, t);
  return &insts_.back();
}

static std::string attrKey(const Attr &a) {
  return std::to_string(a.cls) + ":" + std::to_string(a.code) + ":" + std::to_string(a.value) + ":" + a.key +
         '\0' + a.val + '\0';
}
const AttrSet *Module::attrSet(std::vector<Attr> attrs) {
  std::sort(attrs.begin(), attrs.end());
  attrs.erase(std::unique(attrs.begin(), attrs.end()), attrs.end());
  std::string k;
  for (const Attr &a : attrs) k += attrKey(a) + ";";
  const AttrSet *&s = attrSetIntern_[k];
  if (!s) {
    attrSets_.push_back(std::make_unique<AttrSet>(std::move(attrs)));
    s = attrSets_.back().get();
  }
  return s;
}
const AttrList *Module::attrList(std::vector<Attr> fn, std::vector<Attr> ret, std::vector<std::vector<Attr>> params) {
  auto al = std::make_unique<AttrList>();
  if (!fn.empty()) al->sets.push_back({AttrList::FunctionIndex, attrSet(std::move(fn))});
  if (!ret.empty()) al->sets.push_back({AttrList::ReturnIndex, attrSet(std::move(ret))});
  for (unsigned i = 0; i < params.size(); ++i)
    if (!params[i].empty()) al->sets.push_back({1 + i, attrSet(std::move(params[i]))});
  if (al->sets.empty()) return nullptr;
  std::string k;
  for (auto &p : al->sets) k += numKey(p.first) + ptrKey(p.second);
  const AttrList *&r = attrListIntern_[k];
  if (!r) {
    attrLists_.push_back(std::move(al));
    r = attrLists_.back().get();
  }
  return r;
}

Metadata *Module::mdString(const std::string &s) {
  Metadata *&m = mdIntern_["S" + s];
  if (!m) {
    mds_.push_back(std::make_unique<Metadata>());
    m = mds_.back().get();
    m->kind = Metadata::String;
    m->str = s;
  }
  return m;
}
Metadata *Module::mdValue(Value *v) {
  Metadata *&m = mdIntern_["V" + ptrKey(v)];
  if (!m) {
    mds_.push_back(std::make_unique<Metadata>());
    m = mds_.back().get();
    m->kind = Metadata::ValueMD;
    m->val = v;
  }
  return m;
}
Metadata *Module::mdNode(std::vector<Metadata *> ops) {
  std::string k = "N";
  for (Metadata *o : ops) k += ptrKey(o);
  Metadata *&m = mdIntern_[k];
  if (!m) {
    mds_.push_back(std::make_unique<Metadata>());
    m = mds_.back().get();
    m->kind = Metadata::Node;
    m->ops = std::move(ops);
  }
  return m;
}
Metadata *Module::mdDistinct(std::vector<Metadata *> ops) {
  mds_.push_back(std::make_unique<Metadata>());
  Metadata *m = mds_.back().get();
  m->kind = Metadata::Node;
  m->ops = std::move(ops);
  m->distinct = true;
  return m;
}

// ================================================================ builder
static Type *indexedType(Type *t, const std::vector<Value *> &idx, size_t from) {
  for (size_t i = from; i < idx.size(); ++i) {
    if (t->kind == Type::Struct) {
      auto *c = static_cast<Constant *>(idx[i]);
      assert(idx[i]->vk == Value::Const);
      t = t->elems[c->ck == Constant::Null ? 0 : c->bits];
    } else {
      t = t->elem;
    }
  }
  return t;
}

Instruction *Builder::gep(Type *src, Value *ptr, std::vector<Value *> idx, bool inbounds, const std::string &name) {
  Type *res = m_.ptrTy(indexedType(src, idx, 1), ptr->type->n);
  Instruction *i = m_.newInst(Op::GEP, res);
  i->srcElemTy = src;
  i->inbounds = inbounds;
  i->ops.push_back(ptr);
  i->ops.insert(i->ops.end(), idx.begin(), idx.end());
  return add(i, name);
}
Instruction *Builder::load(Type *t, Value *ptr, uint32_t align, const std::string &name, bool vol) {
  Instruction *i = m_.newInst(Op::Load, t);
  i->ops = {ptr};
  i->align = align;
  i->isVolatile = vol;
  return add(i, name);
}
Instruction *Builder::store(Value *val, Value *ptr, uint32_t align, bool vol) {
  Instruction *i = m_.newInst(Op::Store, m_.voidTy());
  i->ops = {val, ptr};
  i->align = align;
  i->isVolatile = vol;
  return add(i, "");
}
Instruction *Builder::extractValue(Value *agg, std::vector<uint32_t> idx, const std::string &name) {
  Type *t = agg->type;
  for (uint32_t k : idx) t = t->kind == Type::Struct ? t->elems[k] : t->elem;
  Instruction *i = m_.newInst(Op::ExtractValue, t);
  i->ops = {agg};
  i->idx = std::move(idx);
  return add(i, name);
}
Instruction *Builder::insertValue(Value *agg, Value *v, std::vector<uint32_t> idx, const std::string &name) {
  Instruction *i = m_.newInst(Op::InsertValue, agg->type);
  i->ops = {agg, v};
  i->idx = std::move(idx);
  return add(i, name);
}
Instruction *Builder::extractElement(Value *vec, Value *index, const std::string &name) {
  Instruction *i = m_.newInst(Op::ExtractElement, vec->type->elem);
  i->ops = {vec, index};
  return add(i, name);
}
Instruction *Builder::insertElement(Value *vec, Value *v, Value *index, const std::string &name) {
  Instruction *i = m_.newInst(Op::InsertElement, vec->type);
  i->ops = {vec, v, index};
  return add(i, name);
}
Instruction *Builder::shuffleVector(Value *a, Value *b, Constant *mask, const std::string &name) {
  Instruction *i = m_.newInst(Op::ShuffleVector, m_.vecTy(a->type->elem, mask->type->n));
  i->ops = {a, b, mask};
  return add(i, name);
}
Instruction *Builder::cast(CastOpc opc, Value *v, Type *to, const std::string &name) {
  Instruction *i = m_.newInst(Op::Cast, to);
  i->sub = opc;
  i->ops = {v};
  return add(i, name);
}
Instruction *Builder::binop(BinOpc opc, Value *a, Value *b, uint8_t flags, const std::string &name) {
  Instruction *i = m_.newInst(Op::BinOp, a->type);
  i->sub = opc;
  i->flags = flags;
  i->ops = {a, b};
  return add(i, name);
}
static Type *cmpType(Module &m, Type *t) {
  return t->kind == Type::Vector ? m.vecTy(m.intTy(1), t->n) : m.intTy(1);
}
Instruction *Builder::icmp(uint8_t pred, Value *a, Value *b, const std::string &name) {
  Instruction *i = m_.newInst(Op::Cmp, cmpType(m_, a->type));
  i->sub = pred;
  i->ops = {a, b};
  return add(i, name);
}
Instruction *Builder::fcmp(uint8_t pred, Value *a, Value *b, uint8_t fmf, const std::string &name) {
  Instruction *i = icmp(pred, a, b, name);
  i->flags = fmf;
  return i;
}
Instruction *Builder::select(Value *c, Value *a, Value *b, const std::string &name, uint8_t fmf) {
  Instruction *i = m_.newInst(Op::Select, a->type);
  i->ops = {c, a, b};
  i->flags = fmf;
  return add(i, name);
}
Instruction *Builder::fneg(Value *v, uint8_t fmf, const std::string &name) {
  Instruction *i = m_.newInst(Op::FNeg, v->type);
  i->ops = {v};
  i->flags = fmf;
  return add(i, name);
}
Instruction *Builder::allocaInst(Type *allocated, uint32_t align, const std::string &name, Value *size) {
  Instruction *i = m_.newInst(Op::Alloca, m_.ptrTy(allocated, 0));
  i->srcElemTy = allocated;
  i->align = align;
  i->ops = {size ? size : m_.constInt(m_.intTy(32), 1)};
  return add(i, name);
}
Instruction *Builder::switch_(Value *cond, BasicBlock *defaultDest) {
  Instruction *i = m_.newInst(Op::Switch, m_.voidTy());
  i->ops = {cond};
  i->bbs = {defaultDest};
  return add(i, "");
}
Instruction *Builder::phi(Type *t, std::vector<std::pair<Value *, BasicBlock *>> in, const std::string &name,
                          uint8_t fmf) {
  Instruction *i = m_.newInst(Op::Phi, t);
  i->flags = fmf;
  for (auto &p : in) {
    i->ops.push_back(p.first);
    i->bbs.push_back(p.second);
  }
  return add(i, name);
}
Instruction *Builder::call(Function *callee, const std::vector<Value *> &args, const AttrList *attrs,
                           const std::string &name, uint8_t fmf) {
  Instruction *i = m_.newInst(Op::Call, callee->fty->elem);
  i->srcElemTy = callee->fty;
  i->ops.push_back(callee);
  i->ops.insert(i->ops.end(), args.begin(), args.end());
  i->attrs = attrs;
  i->flags = fmf;
  return add(i, name);
}
Instruction *Builder::br(BasicBlock *d) {
  Instruction *i = m_.newInst(Op::Br, m_.voidTy());
  i->bbs = {d};
  return add(i, "");
}
Instruction *Builder::condBr(Value *c, BasicBlock *t, BasicBlock *f) {
  Instruction *i = m_.newInst(Op::Br, m_.voidTy());
  i->bbs = {t, f};
  i->ops = {c};
  return add(i, "");
}
Instruction *Builder::retVoid() { return add(m_.newInst(Op::Ret, m_.voidTy()), ""); }
Instruction *Builder::ret(Value *v) {
  Instruction *i = m_.newInst(Op::Ret, m_.voidTy());
  i->ops = {v};
  return add(i, "");
}
Instruction *Builder::unreachable() { return add(m_.newInst(Op::Unreachable, m_.voidTy()), ""); }

// ================================================================ bitstream (llvm::BitstreamWriter)
namespace {

struct AbbrevOp {
  enum K : uint8_t { Lit, Fixed, VBR, Array, Char6, Blob } k;
  uint64_t v = 0;
};
using Abbrev = std::vector<AbbrevOp>;
AbbrevOp lit(uint64_t v) { return {AbbrevOp::Lit, v}; }
AbbrevOp fixed(uint64_t w) { return {AbbrevOp::Fixed, w}; }
AbbrevOp vbr(uint64_t w) { return {AbbrevOp::VBR, w}; }
const AbbrevOp kArray{AbbrevOp::Array, 0}, kChar6{AbbrevOp::Char6, 0}, kBlob{AbbrevOp::Blob, 0};

bool isChar6(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_'; }
unsigned encodeChar6(uint64_t c) {
  if (c >= 'a' && c <= 'z') return c - 'a';
  if (c >= 'A' && c <= 'Z') return c - 'A' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '.') return 62;
  if (c == '_') return 63;
  throw std::runtime_error("airbc: not a char6 character");
}
bool isDataSequential(const Constant *c);

unsigned log2Ceil(uint64_t x) { // llvm::Log2_32_Ceil
  unsigned r = 0;
  while ((1ull << r) < x) ++r;
  return r;
}

class BitWriter {
public:
  std::vector<uint8_t> out;

  uint64_t bitNo() const { return out.size() * 8 + bit_; }

  void emit(uint32_t val, unsigned n) {
    if (!n) return;
    assert(n <= 32 && (n == 32 || (val >> n) == 0));
    cur_ |= val << bit_;
    if (bit_ + n < 32) {
      bit_ += n;
      return;
    }
    word(cur_);
    cur_ = bit_ ? val >> (32 - bit_) : 0;
    bit_ = (bit_ + n) & 31;
  }
  void emitVBR(uint64_t val, unsigned n) {
    uint64_t thresh = 1ull << (n - 1);
    while (val >= thresh) {
      emit(uint32_t((val & (thresh - 1)) | thresh), n);
      val >>= n - 1;
    }
    emit(uint32_t(val), n);
  }
  void flushToWord() {
    if (bit_) {
      word(cur_);
      cur_ = 0;
      bit_ = 0;
    }
  }

  // Patch 32 bits at an arbitrary (already flushed) bit position — BackpatchWord.
  void patch32(uint64_t bitPos, uint32_t v) {
    assert(bitPos + 32 <= out.size() * 8);
    for (unsigned i = 0; i < 32; ++i) {
      uint64_t b = bitPos + i;
      uint8_t &byte = out[b >> 3];
      byte = uint8_t((byte & ~(1u << (b & 7))) | (((v >> i) & 1u) << (b & 7)));
    }
  }
  void patch64(uint64_t bitPos, uint64_t v) {
    patch32(bitPos, uint32_t(v));
    patch32(bitPos + 32, uint32_t(v >> 32));
  }

  void enterBlock(unsigned id, unsigned width) {
    emit(1, width_); // ENTER_SUBBLOCK
    emitVBR(id, 8);
    emitVBR(width, 4);
    flushToWord();
    size_t sizePos = out.size();
    word(0);
    scopes_.push_back({width_, std::move(abbrevs_), sizePos});
    width_ = width;
    abbrevs_.clear();
    auto it = blockInfo_.find(id);
    if (it != blockInfo_.end()) abbrevs_ = it->second;
  }
  void exitBlock() {
    emit(0, width_); // END_BLOCK
    flushToWord();
    Scope s = std::move(scopes_.back());
    scopes_.pop_back();
    uint32_t words = uint32_t((out.size() - s.sizePos) / 4 - 1);
    for (int i = 0; i < 4; ++i) out[s.sizePos + i] = uint8_t(words >> (8 * i));
    width_ = s.width;
    abbrevs_ = std::move(s.abbrevs);
  }

  unsigned defineAbbrev(Abbrev a) {
    encodeAbbrev(a);
    abbrevs_.push_back(std::make_shared<Abbrev>(std::move(a)));
    return unsigned(abbrevs_.size()) + 3;
  }
  void enterBlockInfo() {
    enterBlock(0, 2);
    curBID_ = ~0u;
  }
  unsigned defineBlockInfoAbbrev(unsigned bid, Abbrev a) {
    if (curBID_ != bid) {
      emitRecord(1 /*SETBID*/, {bid});
      curBID_ = bid;
    }
    encodeAbbrev(a);
    auto &v = blockInfo_[bid];
    v.push_back(std::make_shared<Abbrev>(std::move(a)));
    return unsigned(v.size()) + 3;
  }

  void emitRecord(unsigned code, const std::vector<uint64_t> &vals, unsigned abbrev = 0) {
    if (!abbrev) {
      emit(3, width_); // UNABBREV_RECORD
      emitVBR(code, 6);
      emitVBR(vals.size(), 6);
      for (uint64_t v : vals) emitVBR(v, 6);
      return;
    }
    all_.clear();
    all_.push_back(code);
    all_.insert(all_.end(), vals.begin(), vals.end());
    emitAbbreviated(abbrev, all_, nullptr, 0);
  }
  // vals[0] is the record code.
  void emitWithAbbrev(unsigned abbrev, const std::vector<uint64_t> &vals, const void *blob = nullptr,
                      size_t blobLen = 0) {
    emitAbbreviated(abbrev, vals, static_cast<const uint8_t *>(blob), blobLen);
  }

private:
  struct Scope {
    unsigned width;
    std::vector<std::shared_ptr<Abbrev>> abbrevs;
    size_t sizePos;
  };
  void word(uint32_t w) {
    size_t n = out.size();
    out.resize(n + 4);
    uint8_t *p = out.data() + n;
    p[0] = uint8_t(w), p[1] = uint8_t(w >> 8), p[2] = uint8_t(w >> 16), p[3] = uint8_t(w >> 24);
  }
  void encodeAbbrev(const Abbrev &a) {
    emit(2, width_); // DEFINE_ABBREV
    emitVBR(a.size(), 5);
    for (const AbbrevOp &op : a) {
      emit(op.k == AbbrevOp::Lit, 1);
      if (op.k == AbbrevOp::Lit) {
        emitVBR(op.v, 8);
        continue;
      }
      static const unsigned enc[] = {0, 1, 2, 3, 4, 5};
      emit(enc[op.k], 3);
      if (op.k == AbbrevOp::Fixed || op.k == AbbrevOp::VBR) emitVBR(op.v, 5);
    }
  }
  void scalar(const AbbrevOp &op, uint64_t v) {
    switch (op.k) {
    case AbbrevOp::Fixed: emit(uint32_t(v), unsigned(op.v)); break;
    case AbbrevOp::VBR: emitVBR(v, unsigned(op.v)); break;
    case AbbrevOp::Char6: emit(encodeChar6(v), 6); break;
    default: throw std::runtime_error("airbc: bad scalar abbrev op");
    }
  }
  void emitAbbreviated(unsigned id, const std::vector<uint64_t> &vals, const uint8_t *blob, size_t blobLen) {
    const Abbrev &a = *abbrevs_.at(id - 4);
    emit(id, width_);
    size_t vi = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      const AbbrevOp &op = a[i];
      if (op.k == AbbrevOp::Lit) {
        if (vi >= vals.size() || vals[vi] != op.v) throw std::runtime_error("airbc: literal mismatch");
        ++vi;
      } else if (op.k == AbbrevOp::Array) {
        const AbbrevOp &el = a[++i];
        emitVBR(vals.size() - vi, 6);
        for (; vi < vals.size(); ++vi) scalar(el, vals[vi]);
      } else if (op.k == AbbrevOp::Blob) {
        emitVBR(blobLen, 6);
        flushToWord();
        out.insert(out.end(), blob, blob + blobLen);
        while (out.size() & 3) out.push_back(0);
      } else {
        scalar(op, vals.at(vi++));
      }
    }
    if (vi != vals.size()) throw std::runtime_error("airbc: too many record operands for abbrev");
  }

  uint32_t cur_ = 0;
  unsigned bit_ = 0, width_ = 2, curBID_ = ~0u;
  std::vector<uint64_t> all_; // scratch for emitRecord
  std::vector<std::shared_ptr<Abbrev>> abbrevs_;
  std::vector<Scope> scopes_;
  std::map<unsigned, std::vector<std::shared_ptr<Abbrev>>> blockInfo_;
};

// ================================================================ ValueEnumerator (LLVM 13)
// Pointer-keyed map to unsigned: open addressing, linear probing, no per-entry allocation. Never iterated, so the
// enumeration order does not depend on it. References into it stay valid only until the next insertion.
template <class K> class PtrMap {
public:
  unsigned *find(const K *k) {
    if (keys_.empty()) return nullptr;
    for (size_t i = slot(k);; i = (i + 1) & mask_) {
      if (keys_[i] == k) return &vals_[i];
      if (!keys_[i]) return nullptr;
    }
  }
  const unsigned *find(const K *k) const { return const_cast<PtrMap *>(this)->find(k); }
  bool count(const K *k) const { return find(k) != nullptr; }
  unsigned &operator[](const K *k) { return insert(k).first; }
  // The value slot for k, and whether it was newly created (with value 0).
  std::pair<unsigned &, bool> insert(const K *k) {
    if ((used_ + 1) * 2 > keys_.size()) grow();
    size_t i = slot(k);
    while (keys_[i] && keys_[i] != k) i = (i + 1) & mask_;
    bool fresh = !keys_[i];
    if (fresh) keys_[i] = k, vals_[i] = 0, ++used_;
    return {vals_[i], fresh};
  }
  void clear() {
    if (!used_) return;
    std::fill(keys_.begin(), keys_.end(), nullptr);
    used_ = 0;
  }

private:
  size_t slot(const K *k) const { return size_t((uint64_t(reinterpret_cast<uintptr_t>(k)) >> 4) * 0x9E3779B97F4A7C15ull >> shift_); }
  void grow() {
    std::vector<const K *> oldKeys = std::move(keys_);
    std::vector<unsigned> oldVals = std::move(vals_);
    size_t n = oldKeys.empty() ? 64 : oldKeys.size() * 2;
    keys_.assign(n, nullptr);
    vals_.assign(n, 0);
    mask_ = n - 1;
    shift_ = 64 - unsigned(__builtin_ctzll(n));
    for (size_t j = 0; j < oldKeys.size(); ++j)
      if (oldKeys[j]) {
        size_t i = slot(oldKeys[j]);
        while (keys_[i]) i = (i + 1) & mask_;
        keys_[i] = oldKeys[j], vals_[i] = oldVals[j];
      }
  }
  std::vector<const K *> keys_;
  std::vector<unsigned> vals_;
  size_t used_ = 0, mask_ = 0;
  unsigned shift_ = 64;
};

class Enumerator {
public:
  std::vector<const Type *> types;
  std::vector<std::pair<const Value *, unsigned>> values; // value, use count
  std::vector<const Metadata *> mds;                      // after organize: strings, then the rest
  unsigned numMDStrings = 0;
  std::vector<const AttrList *> attrLists;
  std::vector<std::pair<uint32_t, const AttrSet *>> attrGroups;
  unsigned numModuleValues = 0, cstStart = 0, cstEnd = 0;
  std::vector<const BasicBlock *> blocks;

  Enumerator(const Module &M, bool optimize) : optimize_(optimize) {
    for (const GlobalVariable *G : M.globals) {
      enumValue(G);
      enumType(G->valueType);
    }
    for (const Function *F : M.functions) {
      enumValue(F);
      enumType(F->fty);
      enumAttributes(F->attrs);
    }
    unsigned firstConstant = unsigned(values.size());
    for (const GlobalVariable *G : M.globals)
      if (G->init) enumValue(G->init);
    enumType(M.metadataTy());
    // EnumerateValueSymbolTable only re-touches globals (use counts).
    for (const GlobalVariable *G : M.globals)
      if (!G->name.empty()) enumValue(G);
    for (const Function *F : M.functions)
      if (!F->name.empty()) enumValue(F);
    for (const NamedMD &n : M.namedMD)
      for (const Metadata *md : n.ops) enumMetadata(md);
    for (const Function *F : M.functions) {
      for (const Argument *A : F->args) enumType(A->type);
      for (const BasicBlock *BB : F->blocks)
        for (const Instruction *I : BB->insts) {
          for (const Value *op : I->ops) enumOperandType(op);
          if (I->op == Op::Br || I->op == Op::Switch) // successors are operands (not of a phi)
            for (const BasicBlock *bb : I->bbs) enumType(bb->type);
          if (I->op == Op::GEP || I->op == Op::Alloca) enumType(I->srcElemTy);
          enumType(I->type);
          if (I->op == Op::Call) enumAttributes(I->attrs);
        }
    }
    optimizeConstants(firstConstant, unsigned(values.size()));
    organizeMetadata();
  }

  // Ids are cached in the objects' tag fields (id + 1) and trusted only when the id list holds the object at that
  // position, so a stale tag never answers wrongly; the maps are the fallback and the authority.
  unsigned typeID(const Type *t) const {
    if (unsigned k = t->tag - 1; k < types.size() && types[k] == t) return k;
    const unsigned *it = typeMap_.find(t);
    if (!it || *it == ~0u) throw std::runtime_error("airbc: type not enumerated");
    return *it - 1;
  }
  unsigned valueID(const Value *v) const {
    unsigned k = v->tag - 1;
    if (v->vk == Value::Block ? k < blocks.size() && blocks[k] == v : k < values.size() && values[k].first == v) return k;
    // Function-local values and blocks exist only in the id lists and their tags; only module-level values have a map.
    if (v->vk == Value::Block) throw std::out_of_range("airbc: block not enumerated");
    return mapAt(valueMap_, v) - 1;
  }
  template <class K> static unsigned mapAt(const PtrMap<K> &m, const K *k) {
    const unsigned *it = m.find(k);
    if (!it) throw std::out_of_range("airbc: value not enumerated");
    return *it;
  }
  unsigned mdID(const Metadata *m) const { return mapAt(mdMap_, m) - 1; }
  unsigned mdOrNullID(const Metadata *m) const { return m ? mapAt(mdMap_, m) : 0; }
  unsigned attrListID(const AttrList *a) const { return a ? mapAt(attrListMap_, a) : 0; }
  unsigned attrGroupID(uint32_t idx, const AttrSet *s) const { return attrGroupMap_.at({idx, s}); }

  // ValueEnumerator::incorporateFunction
  void incorporate(const Function &F) {
    numModuleValues = unsigned(values.size());
    inFunction_ = true;
    for (const Argument *A : F.args) enumValue(A);
    cstStart = unsigned(values.size());
    for (const BasicBlock *BB : F.blocks) {
      for (const Instruction *I : BB->insts)
        for (const Value *op : I->ops)
          if (op->vk == Value::Const) enumValue(op);
      blocks.push_back(BB);
      BB->tag = unsigned(blocks.size());
    }
    optimizeConstants(cstStart, unsigned(values.size()));
    enumAttributes(F.attrs);
    cstEnd = unsigned(values.size());
    for (const BasicBlock *BB : F.blocks)
      for (const Instruction *I : BB->insts)
        if (!I->type->isVoid()) enumValue(I);
  }
  void purge() {
    values.resize(numModuleValues);
    blocks.clear();
    inFunction_ = false;
  }

private:
  void enumType(const Type *t) {
    if (unsigned k = t->tag - 1; k < types.size() && types[k] == t) return;
    unsigned &id = typeMap_[t];
    if (id) return;
    if (t->kind == Type::Struct && !t->literal) id = ~0u;
    switch (t->kind) {
    case Type::Pointer:
    case Type::Vector:
    case Type::Array: enumType(t->elem); break;
    case Type::Function:
      enumType(t->elem);
      for (const Type *p : t->elems) enumType(p);
      break;
    case Type::Struct:
      for (const Type *e : t->elems) enumType(e);
      break;
    default: break;
    }
    unsigned &id2 = typeMap_[t];
    if (id2 && id2 != ~0u) return;
    types.push_back(t);
    id2 = unsigned(types.size());
    t->tag = id2;
  }

  // Whether v is in the module-level map.
  bool isModuleLevel(const Value *v) const {
    unsigned k = v->tag - 1;
    if (k < values.size() && values[k].first == v) return !inFunction_ || k < numModuleValues;
    return valueMap_.count(v);
  }
  void enumValue(const Value *v) {
    if (unsigned k = v->tag - 1; k < values.size() && values[k].first == v) {
      values[k].second++;
      return;
    }
    enumType(v->type);
    if (v->vk == Value::Const) {
      auto *c = static_cast<const Constant *>(v);
      if (!c->ops.empty() && !isDataSequential(c)) {
        for (const Constant *op : c->ops) enumValue(op);
        addValue(v);
        return;
      }
    }
    addValue(v);
  }
  void addValue(const Value *v) {
    values.push_back({v, 1});
    v->tag = unsigned(values.size());
    if (!inFunction_) valueMap_[v] = unsigned(values.size());
  }

  void enumOperandType(const Value *v) {
    enumType(v->type);
    if (v->vk != Value::Const || isModuleLevel(v)) return;
    auto *c = static_cast<const Constant *>(v);
    if (isDataSequential(c)) return;
    for (const Constant *op : c->ops) enumOperandType(op);
    if (c->ck == Constant::Expr && c->eop == Constant::CE_GEP) enumType(c->srcElemTy);
  }

  // ValueEnumerator::EnumerateMetadata: post-order over uniqued subgraphs, distinct nodes delayed.
  void enumMetadata(const Metadata *md) {
    std::vector<const Metadata *> delayed;
    std::vector<std::pair<const Metadata *, size_t>> work;
    if (const Metadata *n = enumMDImpl(md)) work.push_back({n, 0});
    while (!work.empty()) {
      const Metadata *n = work.back().first;
      size_t i = work.back().second;
      const Metadata *next = nullptr;
      for (; i < n->ops.size(); ++i)
        if ((next = enumMDImpl(n->ops[i]))) break;
      if (next) {
        work.back().second = i + 1;
        if (next->distinct && !n->distinct)
          delayed.push_back(next);
        else
          work.push_back({next, 0});
        continue;
      }
      work.pop_back();
      mds.push_back(n);
      mdMap_[n] = unsigned(mds.size());
      if (work.empty() || work.back().first->distinct) {
        for (const Metadata *d : delayed) work.push_back({d, 0});
        delayed.clear();
      }
    }
  }
  const Metadata *enumMDImpl(const Metadata *md) {
    if (!md) return nullptr;
    auto ins = mdMap_.insert(md);
    if (!ins.second) return nullptr;
    if (md->kind == Metadata::Node) return md;
    mds.push_back(md);
    ins.first = unsigned(mds.size());
    if (md->kind == Metadata::ValueMD) enumValue(md->val);
    return nullptr;
  }
  // ValueEnumerator::organizeMetadata (module-level metadata only).
  void organizeMetadata() {
    auto order = [](const Metadata *m) {
      if (m->kind == Metadata::String) return 0;
      if (m->kind != Metadata::Node) return 1;
      return m->distinct ? 2 : 3;
    };
    std::stable_sort(mds.begin(), mds.end(),
                     [&](const Metadata *a, const Metadata *b) { return order(a) < order(b); });
    for (unsigned i = 0; i < mds.size(); ++i) {
      mdMap_[mds[i]] = i + 1;
      if (mds[i]->kind == Metadata::String) ++numMDStrings;
    }
  }

  void enumAttributes(const AttrList *al) {
    if (!al) return;
    unsigned &e = attrListMap_[al];
    if (!e) {
      attrLists.push_back(al);
      e = unsigned(attrLists.size());
    }
    for (auto &p : al->sets) {
      unsigned &g = attrGroupMap_[p];
      if (!g) {
        attrGroups.push_back(p);
        g = unsigned(attrGroups.size());
      }
    }
  }

  // ValueEnumerator::OptimizeConstants
  void optimizeConstants(unsigned b, unsigned e) {
    if (!optimize_ || b == e || b + 1 == e) return;
    std::stable_sort(values.begin() + b, values.begin() + e, [&](const auto &l, const auto &r) {
      if (l.first->type != r.first->type) return typeID(l.first->type) < typeID(r.first->type);
      return l.second > r.second;
    });
    std::stable_partition(values.begin() + b, values.begin() + e,
                          [](const auto &p) { return p.first->type->isIntOrIntVector(); });
    for (unsigned i = b; i < e; ++i) {
      values[i].first->tag = i + 1;
      if (!inFunction_) valueMap_[values[i].first] = i + 1;
    }
  }

  bool optimize_, inFunction_ = false;
  PtrMap<Type> typeMap_;
  PtrMap<Value> valueMap_;
  PtrMap<Metadata> mdMap_;
  PtrMap<AttrList> attrListMap_;
  std::map<std::pair<uint32_t, const AttrSet *>, unsigned> attrGroupMap_;
};

// llvm::StringTableBuilder (RAW): append-only, deduplicated.
class StrtabBuilder {
public:
  std::string data;
  uint64_t add(const std::string &s) {
    auto it = map_.find(s);
    if (it != map_.end()) return it->second;
    uint64_t off = data.size();
    map_[s] = off;
    data += s;
    return off;
  }

private:
  std::unordered_map<std::string, uint64_t> map_;
};

// Iteration order of an llvm::StringMap (LLVM 13: djbHash seed 0, quadratic probing, grow at 3/4).
std::vector<size_t> stringMapOrder(const std::vector<std::string> &keys) {
  std::vector<int> table;
  std::vector<uint32_t> hashes;
  size_t items = 0;
  auto hash = [](const std::string &s) {
    uint32_t h = 0;
    for (unsigned char c : s) h = (h << 5) + h + c;
    return h;
  };
  auto place = [](std::vector<int> &t, std::vector<uint32_t> &hs, uint32_t h, int v) {
    size_t n = t.size(), b = h & (n - 1), probe = 1;
    while (t[b] != -1) b = (b + probe++) & (n - 1);
    t[b] = v;
    hs[b] = h;
  };
  for (size_t k = 0; k < keys.size(); ++k) {
    if (table.empty()) {
      table.assign(16, -1);
      hashes.assign(16, 0);
    }
    place(table, hashes, hash(keys[k]), int(k));
    ++items;
    size_t n = table.size(), newSize = 0;
    if (items * 4 > n * 3)
      newSize = n * 2;
    else if (n - items <= n / 8)
      newSize = n;
    if (newSize) {
      std::vector<int> nt(newSize, -1);
      std::vector<uint32_t> nh(newSize, 0);
      for (size_t i = 0; i < n; ++i)
        if (table[i] != -1) place(nt, nh, hashes[i], table[i]);
      table.swap(nt);
      hashes.swap(nh);
    }
  }
  std::vector<size_t> order;
  for (int v : table)
    if (v != -1) order.push_back(size_t(v));
  return order;
}

// ConstantDataSequential: vector/array whose elements are all plain i8/i16/i32/i64/half/float/double values.
bool isDataSequential(const Constant *c) {
  if (c->type->kind != Type::Vector && c->type->kind != Type::Array) return false;
  const Type *e = c->type->elem;
  bool okTy = (e->kind == Type::Int && (e->n == 8 || e->n == 16 || e->n == 32 || e->n == 64)) ||
              e->kind == Type::Half || e->kind == Type::Float || e->kind == Type::Double;
  if (!okTy) return false;
  for (const Constant *op : c->ops)
    if (op->ck != Constant::Int && op->ck != Constant::FP && op->ck != Constant::Null) return false;
  return true;
}

enum StringEncoding { SE_Char6, SE_Fixed7, SE_Fixed8 };
StringEncoding stringEncoding(const std::string &s) {
  bool c6 = true;
  for (char c : s) {
    if (c6) c6 = isChar6(c);
    if ((unsigned char)c & 128) return SE_Fixed8;
  }
  return c6 ? SE_Char6 : SE_Fixed7;
}

std::vector<uint64_t> bytesU(const std::string &s) {
  std::vector<uint64_t> r;
  for (unsigned char c : s) r.push_back(c);
  return r;
}

// Block / record IDs (LLVMBitCodes.h)
enum : unsigned {
  BLOCKINFO_BLOCK_ID = 0, MODULE_BLOCK_ID = 8, PARAMATTR_BLOCK_ID = 9, PARAMATTR_GROUP_BLOCK_ID = 10,
  CONSTANTS_BLOCK_ID = 11, FUNCTION_BLOCK_ID = 12, IDENTIFICATION_BLOCK_ID = 13, VALUE_SYMTAB_BLOCK_ID = 14,
  METADATA_BLOCK_ID = 15, TYPE_BLOCK_ID_NEW = 17, STRTAB_BLOCK_ID = 23, METADATA_KIND_BLOCK_ID = 22,
  OPERAND_BUNDLE_TAGS_BLOCK_ID = 21, SYMTAB_BLOCK_ID = 25, SYNC_SCOPE_NAMES_BLOCK_ID = 26,
};

// ================================================================ ModuleBitcodeWriter (LLVM 13)
class ModuleWriter {
public:
  ModuleWriter(const Module &m, const WriterOptions &o) : M(m), O(o), VE(m, o.optimizeConstants) {}

  std::vector<uint8_t> write() {
    // BitcodeWriter::writeBitcodeHeader
    S.emit('B', 8);
    S.emit('C', 8);
    S.emit(0x0, 4);
    S.emit(0xC, 4);
    S.emit(0xE, 4);
    S.emit(0xD, 4);
    bitcodeStartBit_ = S.bitNo();
    writeIdentificationBlock();
    writeModule();
    if (O.symtab) writeSymtab();
    writeStrtab();
    S.flushToWord();
    std::vector<uint8_t> bc = std::move(S.out);
    if (!O.darwinWrapper) return bc;
    // emitDarwinBCHeaderAndTrailer
    std::vector<uint8_t> r(20);
    uint32_t hdr[5] = {0x0B17C0DEu, 0, 20, uint32_t(bc.size()), O.cpuType};
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 4; ++j) r[4 * i + j] = uint8_t(hdr[i] >> (8 * j));
    r.insert(r.end(), bc.begin(), bc.end());
    while (r.size() & 15) r.push_back(0);
    return r;
  }

private:
  const Module &M;
  const WriterOptions &O;
  Enumerator VE;
  BitWriter S;
  StrtabBuilder strtab_;
  uint64_t bitcodeStartBit_ = 0, vstOffsetPlaceholder_ = 0;
  std::map<const Function *, uint64_t> fnIndex_;
  std::vector<uint64_t> scratch_; // record operands of writeInstruction

  // function-block abbrev IDs from BLOCKINFO
  enum : unsigned {
    VST_ENTRY_8 = 4, VST_ENTRY_7, VST_ENTRY_6, VST_BBENTRY_6,
    CONSTANTS_SETTYPE = 4, CONSTANTS_INTEGER, CONSTANTS_CE_CAST, CONSTANTS_NULL,
    FN_LOAD = 4, FN_UNOP, FN_UNOP_FLAGS, FN_BINOP, FN_BINOP_FLAGS, FN_CAST, FN_RET_VOID, FN_RET_VAL,
    FN_UNREACHABLE, FN_GEP,
  };

  void writeStringRecord(unsigned code, const std::string &s, unsigned abbrev) {
    for (char c : s)
      if (abbrev && !isChar6(c)) abbrev = 0;
    S.emitRecord(code, bytesU(s), abbrev);
  }

  void writeIdentificationBlock() {
    S.enterBlock(IDENTIFICATION_BLOCK_ID, 5);
    unsigned strAbbrev = S.defineAbbrev({lit(1), kArray, kChar6});
    writeStringRecord(1, O.identification, strAbbrev);
    unsigned epochAbbrev = S.defineAbbrev({lit(2), vbr(6)});
    S.emitRecord(2, {0}, epochAbbrev);
    S.exitBlock();
  }

  void writeModule() {
    S.enterBlock(MODULE_BLOCK_ID, 3);
    S.emitRecord(1 /*VERSION*/, {2});
    writeBlockInfo();
    writeTypeTable();
    writeAttributeGroupTable();
    writeAttributeTable();
    writeModuleInfo();
    writeModuleConstants();
    writeModuleMetadataKinds();
    writeModuleMetadata();
    writeNameBlock(OPERAND_BUNDLE_TAGS_BLOCK_ID, 3, O.bundleTags);
    writeNameBlock(SYNC_SCOPE_NAMES_BLOCK_ID, 2, O.syncScopes);
    for (const Function *F : M.functions)
      if (!F->isDeclaration()) writeFunction(*F);
    writeGlobalValueSymbolTable();
    S.exitBlock();
  }

  void writeBlockInfo() {
    unsigned typeBits = log2Ceil(VE.types.size() + 1);
    S.enterBlockInfo();
    S.defineBlockInfoAbbrev(VALUE_SYMTAB_BLOCK_ID, {fixed(3), vbr(8), kArray, fixed(8)});
    S.defineBlockInfoAbbrev(VALUE_SYMTAB_BLOCK_ID, {lit(1), vbr(8), kArray, fixed(7)});
    S.defineBlockInfoAbbrev(VALUE_SYMTAB_BLOCK_ID, {lit(1), vbr(8), kArray, kChar6});
    S.defineBlockInfoAbbrev(VALUE_SYMTAB_BLOCK_ID, {lit(2), vbr(8), kArray, kChar6});
    S.defineBlockInfoAbbrev(CONSTANTS_BLOCK_ID, {lit(1), fixed(typeBits)});
    S.defineBlockInfoAbbrev(CONSTANTS_BLOCK_ID, {lit(4), vbr(8)});
    S.defineBlockInfoAbbrev(CONSTANTS_BLOCK_ID, {lit(11), fixed(4), fixed(typeBits), vbr(8)});
    S.defineBlockInfoAbbrev(CONSTANTS_BLOCK_ID, {lit(2)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(20), vbr(6), fixed(typeBits), vbr(4), fixed(1)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(56), vbr(6), fixed(4)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(56), vbr(6), fixed(4), fixed(8)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(2), vbr(6), vbr(6), fixed(4)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(2), vbr(6), vbr(6), fixed(4), fixed(8)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(3), vbr(6), fixed(typeBits), fixed(4)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(10)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(10), vbr(6)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(15)});
    S.defineBlockInfoAbbrev(FUNCTION_BLOCK_ID, {lit(43), fixed(1), fixed(typeBits), kArray, vbr(6)});
    S.exitBlock();
  }

  void writeTypeTable() {
    unsigned nb = log2Ceil(VE.types.size() + 1);
    S.enterBlock(TYPE_BLOCK_ID_NEW, 4);
    unsigned ptrAbbrev = S.defineAbbrev({lit(8), fixed(nb), lit(0)});
    S.defineAbbrev({lit(25), lit(0)}); // opaque pointer (unused: AIR is typed)
    unsigned fnAbbrev = S.defineAbbrev({lit(21), fixed(1), kArray, fixed(nb)});
    unsigned anonAbbrev = S.defineAbbrev({lit(18), fixed(1), kArray, fixed(nb)});
    unsigned nameAbbrev = S.defineAbbrev({lit(19), kArray, kChar6});
    unsigned namedAbbrev = S.defineAbbrev({lit(20), fixed(1), kArray, fixed(nb)});
    unsigned arrAbbrev = S.defineAbbrev({lit(11), vbr(8), fixed(nb)});
    S.emitRecord(1 /*NUMENTRY*/, {VE.types.size()});
    for (const Type *t : VE.types) {
      std::vector<uint64_t> v;
      unsigned code = 0, abbrev = 0;
      switch (t->kind) {
      case Type::Void: code = 2; break;
      case Type::Half: code = 10; break;
      case Type::Float: code = 3; break;
      case Type::Double: code = 4; break;
      case Type::Label: code = 5; break;
      case Type::Metadata: code = 16; break;
      case Type::Int: code = 7; v = {t->n}; break;
      case Type::Pointer:
        code = 8;
        v = {VE.typeID(t->elem), t->n};
        if (t->n == 0) abbrev = ptrAbbrev;
        break;
      case Type::Function:
        code = 21;
        v = {t->vararg, VE.typeID(t->elem)};
        for (const Type *p : t->elems) v.push_back(VE.typeID(p));
        abbrev = fnAbbrev;
        break;
      case Type::Struct:
        v.push_back(t->packed);
        for (const Type *e : t->elems) v.push_back(VE.typeID(e));
        if (t->literal) {
          code = 18;
          abbrev = anonAbbrev;
        } else {
          if (t->opaque) {
            code = 6;
          } else {
            code = 20;
            abbrev = namedAbbrev;
          }
          if (!t->name.empty()) writeStringRecord(19, t->name, nameAbbrev);
        }
        break;
      case Type::Array:
        code = 11;
        v = {t->n, VE.typeID(t->elem)};
        abbrev = arrAbbrev;
        break;
      case Type::Vector: code = 12; v = {t->n, VE.typeID(t->elem)}; break;
      }
      S.emitRecord(code, v, abbrev);
    }
    S.exitBlock();
  }

  void writeAttributeGroupTable() {
    if (VE.attrGroups.empty()) return;
    S.enterBlock(PARAMATTR_GROUP_BLOCK_ID, 3);
    for (unsigned g = 0; g < VE.attrGroups.size(); ++g) {
      auto &p = VE.attrGroups[g];
      std::vector<uint64_t> r = {g + 1, p.first};
      for (const Attr &a : *p.second) {
        if (a.cls == Attr::Enum) {
          r.push_back(0);
          r.push_back(a.code);
        } else if (a.cls == Attr::Int) {
          r.push_back(1);
          r.push_back(a.code);
          r.push_back(a.value);
        } else {
          r.push_back(a.cls);
          for (unsigned char c : a.key) r.push_back(c);
          r.push_back(0);
          if (a.cls == Attr::StrVal) {
            for (unsigned char c : a.val) r.push_back(c);
            r.push_back(0);
          }
        }
      }
      S.emitRecord(3 /*PARAMATTR_GRP_CODE_ENTRY*/, r);
    }
    S.exitBlock();
  }

  void writeAttributeTable() {
    if (VE.attrLists.empty()) return;
    S.enterBlock(PARAMATTR_BLOCK_ID, 3);
    for (const AttrList *al : VE.attrLists) {
      std::vector<uint64_t> r;
      for (auto &p : al->sets) r.push_back(VE.attrGroupID(p.first, p.second));
      S.emitRecord(2 /*PARAMATTR_CODE_ENTRY*/, r);
    }
    S.exitBlock();
  }

  void writeModuleInfo() {
    if (!M.triple.empty()) writeStringRecord(2, M.triple, 0);
    if (!M.datalayout.empty()) writeStringRecord(3, M.datalayout, 0);
    unsigned simpleGVarAbbrev = 0;
    // Section names, in order of first use (MODULE_CODE_SECTIONNAME); a global refers to one by 1-based index.
    std::vector<std::string> sections;
    for (const GlobalVariable* G : M.globals)
      if (!G->section.empty() && std::find(sections.begin(), sections.end(), G->section) == sections.end())
        sections.push_back(G->section);
    for (const std::string& name : sections) writeStringRecord(5, name, 0);
    if (!M.globals.empty()) {
      unsigned maxType = 0, maxAlign = 0;
      for (const GlobalVariable *G : M.globals) {
        maxType = std::max(maxType, VE.typeID(G->valueType));
        maxAlign = std::max(maxAlign, G->align);
      }
      simpleGVarAbbrev = S.defineAbbrev({lit(7), vbr(8), vbr(8), fixed(log2Ceil(maxType + 1)), vbr(6), vbr(6),
                                         fixed(5), maxAlign ? fixed(log2Ceil(encodedAlign(maxAlign) + 1)) : lit(0),
                                         lit(0)});
    }
    {
      StringEncoding e = stringEncoding(M.sourceFileName);
      AbbrevOp op = e == SE_Char6 ? kChar6 : e == SE_Fixed7 ? fixed(7) : fixed(8);
      unsigned a = S.defineAbbrev({lit(16), kArray, op});
      S.emitRecord(16, bytesU(M.sourceFileName), a);
    }
    for (const GlobalVariable *G : M.globals) {
      uint64_t section = 0;
      if (!G->section.empty())
        section = 1 + (std::find(sections.begin(), sections.end(), G->section) - sections.begin());
      std::vector<uint64_t> v = {strtab_.add(G->name),
                                 G->name.size(),
                                 VE.typeID(G->valueType),
                                 uint64_t(G->type->n) << 2 | 2 | G->isConstant,
                                 G->init ? VE.valueID(G->init) + 1 : 0,
                                 G->linkage,
                                 encodedAlign(G->align),
                                 section};
      bool dsoLocal = isLocalLinkage(G->linkage);
      if (G->unnamedAddr || dsoLocal || section || G->externallyInitialized) {
        v.insert(v.end(), {0 /*visibility*/, 0 /*tls*/, G->unnamedAddr, G->externallyInitialized, 0 /*dll*/,
                           0 /*comdat*/, 0 /*attrs*/, dsoLocal, strtab_.add(""), 0 /*partition*/});
        S.emitRecord(7 /*MODULE_CODE_GLOBALVAR*/, v);
      } else {
        S.emitRecord(7, v, simpleGVarAbbrev);
      }
    }
    for (const Function *F : M.functions) {
      std::vector<uint64_t> v = {
          strtab_.add(F->name), F->name.size(), VE.typeID(F->fty), F->callingConv, F->isDeclaration(),
          F->linkage, VE.attrListID(F->attrs), 0 /*align*/, 0 /*section*/, 0 /*visibility*/, 0 /*gc*/,
          F->unnamedAddr, 0 /*prologue*/, 0 /*dllstorage*/, 0 /*comdat*/, 0 /*prefix*/, 0 /*personality*/,
          isLocalLinkage(F->linkage), 0 /*addrspace*/, strtab_.add(""), 0 /*partition size*/};
      S.emitRecord(8 /*MODULE_CODE_FUNCTION*/, v);
    }
    if (M.functions.empty()) return;
    // writeValueSymbolTableForwardDecl
    unsigned a = S.defineAbbrev({lit(13), fixed(32)});
    S.emitWithAbbrev(a, {13, 0});
    vstOffsetPlaceholder_ = S.bitNo() - 32;
  }

  static bool isLocalLinkage(uint32_t l) { return l == Link_Internal || l == Link_Private; }

  void writeModuleConstants() {
    for (unsigned i = 0; i < VE.values.size(); ++i)
      if (VE.values[i].first->vk != Value::Func && VE.values[i].first->vk != Value::Global) {
        writeConstants(i, unsigned(VE.values.size()), true);
        return;
      }
  }

  void writeConstants(unsigned first, unsigned last, bool isGlobal) {
    if (first == last) return;
    S.enterBlock(CONSTANTS_BLOCK_ID, 4);
    unsigned aggAbbrev = 0, str8Abbrev = 0, cstr7Abbrev = 0, cstr6Abbrev = 0;
    if (isGlobal) {
      aggAbbrev = S.defineAbbrev({lit(7), kArray, fixed(log2Ceil(last + 1))});
      str8Abbrev = S.defineAbbrev({lit(8), kArray, fixed(8)});
      cstr7Abbrev = S.defineAbbrev({lit(9), kArray, fixed(7)});
      cstr6Abbrev = S.defineAbbrev({lit(9), kArray, kChar6});
    }
    const Type *lastTy = nullptr;
    for (unsigned i = first; i < last; ++i) {
      const Value *v = VE.values[i].first;
      if (v->type != lastTy) {
        lastTy = v->type;
        S.emitRecord(1 /*SETTYPE*/, {VE.typeID(lastTy)}, CONSTANTS_SETTYPE);
      }
      auto *c = static_cast<const Constant *>(v);
      switch (c->ck) {
      case Constant::Null: S.emitRecord(2, {}); break;
      case Constant::Undef: S.emitRecord(3, {}); break;
      case Constant::Poison: S.emitRecord(26, {}); break;
      case Constant::Int: {
        unsigned w = c->type->n;
        int64_t sv = w >= 64 ? int64_t(c->bits) : int64_t(c->bits << (64 - w)) >> (64 - w);
        uint64_t enc = sv >= 0 ? uint64_t(sv) << 1 : (uint64_t(-sv) << 1) | 1;
        S.emitRecord(4, {enc}, CONSTANTS_INTEGER);
        break;
      }
      case Constant::FP: S.emitRecord(6, {c->bits}); break;
      case Constant::GlobalVar: throw std::runtime_error("airbc: global in constant block");
      case Constant::Expr: {
        const Constant *a = c->ops[0];
        switch (c->eop) {
        case Constant::CE_Cast:
          S.emitRecord(11 /*CE_CAST*/, {c->sub, VE.typeID(a->type), VE.valueID(a)}, CONSTANTS_CE_CAST);
          break;
        case Constant::CE_BinOp: {
          std::vector<uint64_t> r = {c->sub, VE.valueID(a), VE.valueID(c->ops[1])};
          if (c->flags) r.push_back(c->flags);
          S.emitRecord(10 /*CE_BINOP*/, r);
          break;
        }
        case Constant::CE_GEP: {
          std::vector<uint64_t> r = {VE.typeID(c->srcElemTy)};
          for (const Constant *op : c->ops) {
            r.push_back(VE.typeID(op->type));
            r.push_back(VE.valueID(op));
          }
          S.emitRecord(c->inbounds ? 20 /*CE_INBOUNDS_GEP*/ : 12 /*CE_GEP*/, r);
          break;
        }
        }
        break;
      }
      case Constant::Aggregate: {
        std::vector<uint64_t> r;
        if (!isDataSequential(c)) {
          for (const Constant *op : c->ops) r.push_back(VE.valueID(op));
          S.emitRecord(7, r, aggAbbrev);
        } else if (c->type->kind == Type::Array && c->type->elem->kind == Type::Int && c->type->elem->n == 8) {
          // ConstantDataArray::isString: CSTRING when NUL-terminated with no interior NUL, else STRING.
          size_t n = c->ops.size();
          bool cstr = n && c->ops[n - 1]->ck == Constant::Null;
          for (size_t k = 0; cstr && k + 1 < n; ++k) cstr = c->ops[k]->ck != Constant::Null;
          if (cstr) --n;
          bool c7 = cstr, c6 = cstr;
          for (size_t k = 0; k < n; ++k) {
            uint64_t ch = c->ops[k]->bits;
            r.push_back(ch);
            c7 &= (ch & 128) == 0;
            if (c6) c6 = isChar6(char(ch));
          }
          S.emitRecord(cstr ? 9 : 8, r, c6 ? cstr6Abbrev : c7 ? cstr7Abbrev : cstr ? 0 : str8Abbrev);
        } else {
          for (const Constant *op : c->ops) r.push_back(op->bits); // CST_CODE_DATA: raw element values
          S.emitRecord(22, r);
        }
        break;
      }
      }
    }
    S.exitBlock();
  }

  void writeModuleMetadataKinds() {
    if (O.mdKinds.empty()) return;
    S.enterBlock(METADATA_KIND_BLOCK_ID, 3);
    for (unsigned i = 0; i < O.mdKinds.size(); ++i) {
      std::vector<uint64_t> r = {i};
      for (unsigned char c : O.mdKinds[i]) r.push_back(c);
      S.emitRecord(6 /*METADATA_KIND*/, r);
    }
    S.exitBlock();
  }

  void writeModuleMetadata() {
    if (VE.mds.empty() && M.namedMD.empty()) return;
    S.enterBlock(METADATA_BLOCK_ID, 4);
    S.defineAbbrev({lit(7), fixed(1), vbr(6), vbr(8), vbr(6), vbr(6), fixed(1)});     // DILocation
    S.defineAbbrev({lit(12), fixed(1), vbr(6), fixed(1), vbr(6), kArray, vbr(6)});    // GenericDINode
    unsigned offsetAbbrev = S.defineAbbrev({lit(38), fixed(32), fixed(32)});
    unsigned indexAbbrev = S.defineAbbrev({lit(39), kArray, vbr(6)});

    // writeMetadataStrings
    if (VE.numMDStrings) {
      BitWriter w;
      std::string blob;
      for (unsigned i = 0; i < VE.numMDStrings; ++i) w.emitVBR(VE.mds[i]->str.size(), 6);
      w.flushToWord();
      blob.assign(w.out.begin(), w.out.end());
      uint64_t offset = blob.size();
      for (unsigned i = 0; i < VE.numMDStrings; ++i) blob += VE.mds[i]->str;
      unsigned a = S.defineAbbrev({lit(35), vbr(6), vbr(6), kBlob});
      S.emitWithAbbrev(a, {35, VE.numMDStrings, offset}, blob.data(), blob.size());
    }

    const size_t nonStrings = VE.mds.size() - VE.numMDStrings;
    const bool index = nonStrings > 25; // IndexThreshold
    if (index) S.emitWithAbbrev(offsetAbbrev, {38, 0, 0});
    uint64_t indexOffsetRecordBitPos = S.bitNo();
    std::vector<uint64_t> indexPos;
    for (size_t i = VE.numMDStrings; i < VE.mds.size(); ++i) {
      const Metadata *md = VE.mds[i];
      indexPos.push_back(S.bitNo());
      if (md->kind == Metadata::Node) {
        std::vector<uint64_t> r;
        for (const Metadata *op : md->ops) r.push_back(VE.mdOrNullID(op));
        S.emitRecord(md->distinct ? 5 : 3, r);
      } else {
        S.emitRecord(2 /*METADATA_VALUE*/, {VE.typeID(md->val->type), VE.valueID(md->val)});
      }
    }
    if (index) {
      S.patch64(indexOffsetRecordBitPos - 64, S.bitNo() - indexOffsetRecordBitPos);
      uint64_t prev = indexOffsetRecordBitPos;
      std::vector<uint64_t> r = {39};
      for (uint64_t p : indexPos) {
        r.push_back(p - prev);
        prev = p;
      }
      S.emitWithAbbrev(indexAbbrev, r);
    }
    // writeNamedMetadata
    if (!M.namedMD.empty()) {
      unsigned nameAbbrev = S.defineAbbrev({lit(4), kArray, fixed(8)});
      for (const NamedMD &n : M.namedMD) {
        S.emitRecord(4, bytesU(n.name), nameAbbrev);
        std::vector<uint64_t> r;
        for (const Metadata *op : n.ops) r.push_back(VE.mdID(op));
        S.emitRecord(10 /*NAMED_NODE*/, r);
      }
    }
    S.exitBlock();
  }

  void writeNameBlock(unsigned id, unsigned width, const std::vector<std::string> &names) {
    if (names.empty()) return;
    S.enterBlock(id, width);
    for (const std::string &n : names) S.emitRecord(1, bytesU(n));
    S.exitBlock();
  }

  // ---- instructions
  bool pushValueAndType(const Value *v, unsigned instID, std::vector<uint64_t> &vals) {
    unsigned id = VE.valueID(v);
    vals.push_back(uint32_t(instID - id));
    if (id >= instID) {
      vals.push_back(VE.typeID(v->type));
      return true;
    }
    return false;
  }
  void pushValue(const Value *v, unsigned instID, std::vector<uint64_t> &vals) {
    vals.push_back(uint32_t(instID - VE.valueID(v)));
  }
  void pushValueSigned(const Value *v, unsigned instID, std::vector<uint64_t> &vals) {
    int64_t d = int64_t(instID) - int64_t(VE.valueID(v));
    vals.push_back(d >= 0 ? uint64_t(d) << 1 : (uint64_t(-d) << 1) | 1);
  }
  static unsigned encodedAlign(uint32_t bytes) { return bytes ? log2Ceil(bytes) + 1 : 0; }

  void writeInstruction(const Instruction &I, unsigned instID) {
    std::vector<uint64_t> &v = scratch_;
    v.clear();
    unsigned code = 0, abbrev = 0;
    switch (I.op) {
    case Op::Cast:
      code = 3;
      if (!pushValueAndType(I.ops[0], instID, v)) abbrev = FN_CAST;
      v.push_back(VE.typeID(I.type));
      v.push_back(I.sub);
      break;
    case Op::BinOp:
      code = 2;
      if (!pushValueAndType(I.ops[0], instID, v)) abbrev = FN_BINOP;
      pushValue(I.ops[1], instID, v);
      v.push_back(I.sub);
      if (I.flags) {
        if (abbrev == FN_BINOP) abbrev = FN_BINOP_FLAGS;
        v.push_back(I.flags);
      }
      break;
    case Op::GEP:
      code = 43;
      abbrev = FN_GEP;
      v.push_back(I.inbounds);
      v.push_back(VE.typeID(I.srcElemTy));
      for (const Value *op : I.ops) pushValueAndType(op, instID, v);
      break;
    case Op::ExtractValue:
      code = 26;
      pushValueAndType(I.ops[0], instID, v);
      v.insert(v.end(), I.idx.begin(), I.idx.end());
      break;
    case Op::InsertValue:
      code = 27;
      pushValueAndType(I.ops[0], instID, v);
      pushValueAndType(I.ops[1], instID, v);
      v.insert(v.end(), I.idx.begin(), I.idx.end());
      break;
    case Op::Select:
      code = 29; // VSELECT
      pushValueAndType(I.ops[1], instID, v);
      pushValue(I.ops[2], instID, v);
      pushValueAndType(I.ops[0], instID, v);
      if (I.flags) v.push_back(I.flags);
      break;
    case Op::FNeg:
      code = 56; // UNOP
      if (!pushValueAndType(I.ops[0], instID, v)) abbrev = FN_UNOP;
      v.push_back(0); // UNOP_FNEG
      if (I.flags) {
        if (abbrev == FN_UNOP) abbrev = FN_UNOP_FLAGS;
        v.push_back(I.flags);
      }
      break;
    case Op::Alloca: {
      code = 19;
      unsigned enc = encodedAlign(I.align);
      v.push_back(VE.typeID(I.srcElemTy));
      v.push_back(VE.typeID(I.ops[0]->type));
      v.push_back(VE.valueID(I.ops[0]));
      v.push_back((enc & 31) | (enc >> 5) << 8 | 1u << 6 /*explicit type*/);
      break;
    }
    case Op::Switch:
      code = 12;
      v.push_back(VE.typeID(I.ops[0]->type));
      pushValue(I.ops[0], instID, v);
      v.push_back(VE.valueID(I.bbs[0]));
      for (size_t k = 1; k < I.ops.size(); ++k) {
        v.push_back(VE.valueID(I.ops[k]));
        v.push_back(VE.valueID(I.bbs[k]));
      }
      break;
    case Op::ExtractElement:
      code = 6;
      pushValueAndType(I.ops[0], instID, v);
      pushValueAndType(I.ops[1], instID, v);
      break;
    case Op::InsertElement:
      code = 7;
      pushValueAndType(I.ops[0], instID, v);
      pushValue(I.ops[1], instID, v);
      pushValueAndType(I.ops[2], instID, v);
      break;
    case Op::ShuffleVector:
      code = 8;
      pushValueAndType(I.ops[0], instID, v);
      pushValue(I.ops[1], instID, v);
      pushValue(I.ops[2], instID, v);
      break;
    case Op::Cmp:
      code = 28; // CMP2
      pushValueAndType(I.ops[0], instID, v);
      pushValue(I.ops[1], instID, v);
      v.push_back(I.sub);
      if (I.flags) v.push_back(I.flags);
      break;
    case Op::Ret:
      code = 10;
      if (I.ops.empty())
        abbrev = FN_RET_VOID;
      else if (!pushValueAndType(I.ops[0], instID, v))
        abbrev = FN_RET_VAL;
      break;
    case Op::Br:
      code = 11;
      v.push_back(VE.valueID(I.bbs[0]));
      if (I.bbs.size() > 1) {
        v.push_back(VE.valueID(I.bbs[1]));
        pushValue(I.ops[0], instID, v);
      }
      break;
    case Op::Unreachable:
      code = 15;
      abbrev = FN_UNREACHABLE;
      break;
    case Op::Phi: {
      v.push_back(VE.typeID(I.type));
      for (size_t k = 0; k < I.ops.size(); ++k) {
        pushValueSigned(I.ops[k], instID, v);
        v.push_back(VE.valueID(I.bbs[k]));
      }
      if (I.flags) v.push_back(I.flags);
      S.emitRecord(16, v);
      return;
    }
    case Op::Load:
      code = 20;
      if (!pushValueAndType(I.ops[0], instID, v)) abbrev = FN_LOAD;
      v.push_back(VE.typeID(I.type));
      v.push_back(encodedAlign(I.align));
      v.push_back(I.isVolatile);
      break;
    case Op::Store:
      code = 44;
      pushValueAndType(I.ops[1], instID, v);
      pushValueAndType(I.ops[0], instID, v);
      v.push_back(encodedAlign(I.align));
      v.push_back(I.isVolatile);
      break;
    case Op::Call: {
      code = 34;
      auto *callee = static_cast<const Function *>(I.ops[0]);
      v.push_back(VE.attrListID(I.attrs));
      v.push_back(uint64_t(callee->callingConv) << 1 | uint64_t(I.tail) | 1ull << 15 /*explicit type*/ |
                  uint64_t(I.flags != 0) << 17);
      if (I.flags) v.push_back(I.flags);
      v.push_back(VE.typeID(I.srcElemTy));
      pushValueAndType(I.ops[0], instID, v);
      for (size_t k = 1; k < I.ops.size(); ++k) pushValue(I.ops[k], instID, v);
      break;
    }
    }
    S.emitRecord(code, v, abbrev);
  }

  void writeFunction(const Function &F) {
    fnIndex_[&F] = S.bitNo();
    S.enterBlock(FUNCTION_BLOCK_ID, 4);
    VE.incorporate(F);
    S.emitRecord(1 /*DECLAREBLOCKS*/, {VE.blocks.size()});
    writeConstants(VE.cstStart, VE.cstEnd, false);
    unsigned instID = VE.cstEnd;
    for (const BasicBlock *BB : F.blocks)
      for (const Instruction *I : BB->insts) {
        writeInstruction(*I, instID);
        if (!I->type->isVoid()) ++instID;
      }
    writeFunctionLevelValueSymbolTable(F);
    VE.purge();
    S.exitBlock();
  }

  void writeFunctionLevelValueSymbolTable(const Function &F) {
    // Names in LLParser symbol-table insertion order: arguments; then each block's label at its first mention
    // (a forward-referenced label is created when a branch/phi names it); each instruction's name after the
    // labels it references. Forward-referenced *values* are placeholders outside the table.
    std::vector<const Value *> named;
    std::unordered_map<const Value *, bool> seen;
    auto note = [&](const Value *v) {
      if (!v->name.empty() && !seen[v]) {
        seen[v] = true;
        named.push_back(v);
      }
    };
    for (const Argument *A : F.args) note(A);
    for (const BasicBlock *BB : F.blocks) {
      note(BB);
      for (const Instruction *I : BB->insts) {
        for (const BasicBlock *t : I->bbs) note(t);
        note(I);
      }
    }
    if (named.empty()) return;
    std::vector<std::string> keys;
    for (const Value *v : named) keys.push_back(v->name);
    S.enterBlock(VALUE_SYMTAB_BLOCK_ID, 4);
    for (size_t k : stringMapOrder(keys)) {
      const Value *val = named[k];
      StringEncoding e = stringEncoding(val->name);
      unsigned abbrev = VST_ENTRY_8, code;
      if (val->vk == Value::Block) {
        code = 2;
        if (e == SE_Char6) abbrev = VST_BBENTRY_6;
      } else {
        code = 1;
        if (e == SE_Char6)
          abbrev = VST_ENTRY_6;
        else if (e == SE_Fixed7)
          abbrev = VST_ENTRY_7;
      }
      std::vector<uint64_t> r = {code, VE.valueID(val)};
      for (unsigned char c : val->name) r.push_back(c);
      S.emitWithAbbrev(abbrev, r);
    }
    S.exitBlock();
  }

  void writeGlobalValueSymbolTable() {
    if (M.functions.empty()) return;
    uint64_t off = S.bitNo() - bitcodeStartBit_;
    assert((off & 31) == 0);
    S.patch32(vstOffsetPlaceholder_, uint32_t(off / 32 + 1));
    S.enterBlock(VALUE_SYMTAB_BLOCK_ID, 4);
    unsigned a = S.defineAbbrev({lit(3), vbr(8), vbr(8)});
    for (const Function *F : M.functions) {
      if (F->isDeclaration()) continue;
      uint64_t idx = fnIndex_.at(F) - bitcodeStartBit_;
      S.emitWithAbbrev(a, {3, VE.valueID(F), idx / 32 + 1});
    }
    S.exitBlock();
  }

  // ---- irsymtab::build (storage version 3)
  void writeSymtab() {
    std::vector<uint8_t> blob(76);
    auto put32 = [&](size_t at, uint32_t x) {
      for (int i = 0; i < 4; ++i) blob[at + i] = uint8_t(x >> (8 * i));
    };
    auto app32 = [&](uint32_t x) {
      blob.resize(blob.size() + 4);
      put32(blob.size() - 4, x);
    };
    auto setStr = [&](size_t at, const std::string &s) {
      put32(at, uint32_t(strtab_.add(s)));
      put32(at + 4, uint32_t(s.size()));
    };
    put32(0, 3); // Version
    setStr(4, O.producer);
    setStr(44, M.triple);
    setStr(52, M.sourceFileName);
    // Modules: one {Begin, End, UncBegin}
    put32(12, uint32_t(blob.size()));
    put32(16, 1);
    app32(0);
    app32(uint32_t(M.functions.size() + M.globals.size()));
    app32(0);
    put32(20, uint32_t(blob.size())); // Comdats
    put32(24, 0);
    put32(28, uint32_t(blob.size())); // Symbols, in Module::global_values() order: functions, then globals
    put32(32, uint32_t(M.functions.size() + M.globals.size()));
    auto addSym = [&](const std::string &name, bool executable, bool undefined, uint32_t linkage,
                      uint32_t unnamedAddr) {
      size_t at = blob.size();
      blob.resize(at + 24);
      setStr(at, name);     // Name (no mangling prefix on air64)
      setStr(at + 8, name); // IRName
      put32(at + 16, 0xFFFFFFFFu);
      uint32_t flags = 0;
      if (executable) flags |= 1u << 13;  // FB_executable
      if (undefined) flags |= 1u << 3;    // FB_undefined
      if (!isLocalLinkage(linkage)) flags |= 1u << 10; // FB_global
      if (linkage == Link_Private || name.compare(0, 5, "llvm.") == 0) flags |= 1u << 11; // FB_format_specific
      if (unnamedAddr == 1) flags |= 1u << 12; // FB_unnamed_addr
      put32(at + 20, flags);
    };
    for (const Function *F : M.functions)
      addSym(F->name, true, F->isDeclaration(), F->linkage, F->unnamedAddr);
    for (const GlobalVariable *G : M.globals) addSym(G->name, false, !G->init, G->linkage, G->unnamedAddr);
    put32(36, uint32_t(blob.size())); // Uncommons
    put32(40, 0);
    setStr(60, ""); // COFFLinkerOpts
    put32(68, uint32_t(blob.size())); // DependentLibraries
    put32(72, 0);

    S.enterBlock(SYMTAB_BLOCK_ID, 3);
    unsigned a = S.defineAbbrev({lit(1), kBlob});
    S.emitWithAbbrev(a, {1}, blob.data(), blob.size());
    S.exitBlock();
  }

  void writeStrtab() {
    S.enterBlock(STRTAB_BLOCK_ID, 3);
    unsigned a = S.defineAbbrev({lit(1), kBlob});
    S.emitWithAbbrev(a, {1}, strtab_.data.data(), strtab_.data.size());
    S.exitBlock();
  }
};

} // namespace

std::vector<uint8_t> writeBitcode(const Module &m, const WriterOptions &opts) { return ModuleWriter(m, opts).write(); }

} // namespace airbc
