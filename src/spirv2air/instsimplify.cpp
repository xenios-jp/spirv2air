// O1 instruction simplification: one forward walk over the function, repeated while it finds something. An instruction
// whose result is known (a constant, or an existing value) is recorded in a replacement map; operands are resolved
// through the map as the walk goes and rewritten at the end. Everything is exact: scalar integer arithmetic wraps as
// the instruction does, folds that would be poison (shift by >= the width, division by zero or of INT_MIN by -1) are
// left alone, and no float is computed except by moving bits (fneg of a constant, bitcast of a constant).
#include "spirv2air/instsimplify.h"

#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

uint64_t widthMask(unsigned w) { return w >= 64 ? ~0ull : (1ull << w) - 1; }

int64_t signExtend(uint64_t v, unsigned w) { return w >= 64 ? int64_t(v) : int64_t(v << (64 - w)) >> (64 - w); }

bool isScalarInt(const Type* t) { return t->kind == Type::Int; }

bool isFloat(const Type* t) { return t->kind == Type::Half || t->kind == Type::Float || t->kind == Type::Double; }

// A scalar integer constant (a null one is zero).
bool constInt(const Value* v, uint64_t& out) {
  if (v->vk != Value::Const || !isScalarInt(v->type)) return false;
  const auto* c = static_cast<const Constant*>(v);
  if (c->ck != Constant::Int && c->ck != Constant::Null) return false;
  out = c->ck == Constant::Int ? c->bits : 0;
  return true;
}

class Simplifier {
 public:
  Simplifier(Module& m, Function* f) : m_(m), f_(f) {
    unsigned n = 0;
    for (const BasicBlock* b : f->blocks) n += unsigned(b->insts.size());
    ibase_ = m.allocTags(n);
    n_ = 0;
    for (BasicBlock* b : f->blocks)
      for (Instruction* i : b->insts) i->tag = ibase_ + n_++;
    repl_.assign(n_, nullptr);
  }

  bool run() {
    bool any = false;
    for (bool changed = true; changed;) {
      changed = false;
      for (BasicBlock* b : f_->blocks)
        for (Instruction* i : b->insts) {
          if (repl_[id(i)]) continue;
          for (Value*& v : i->ops) v = resolve(v);
          if (Value* r = simplify(i); r && r != i) repl_[id(i)] = r, changed = true;
        }
      any |= changed;
    }
    if (any)
      for (BasicBlock* b : f_->blocks)
        for (Instruction* i : b->insts)
          for (Value*& v : i->ops) v = resolve(v);
    return any;
  }

 private:
  unsigned id(const Value* v) const {
    unsigned k = v->tag - ibase_;
    return v->vk == Value::Inst && k < n_ ? k : ~0u;
  }
  Value* resolve(Value* v) const {
    for (unsigned k = id(v); k != ~0u && repl_[k]; k = id(v)) v = repl_[k];
    return v;
  }

  Value* cint(Type* t, uint64_t v) { return m_.constInt(t, int64_t(v)); }

  // The bits that may be set in an integer value (a superset), looking a few instructions deep.
  uint64_t maybeBits(const Value* v, int depth = 3) const {
    const unsigned w = v->type->n;
    const uint64_t all = widthMask(w);
    uint64_t c;
    if (constInt(v, c)) return c;
    if (v->vk != Value::Inst || depth == 0) return all;
    const auto* i = static_cast<const Instruction*>(v);
    uint64_t k;
    switch (i->op) {
      case Op::BinOp:
        if (!isScalarInt(i->type)) return all;
        if (i->sub == And) return maybeBits(i->ops[0], depth - 1) & maybeBits(i->ops[1], depth - 1);
        if (i->sub == Or || i->sub == Xor) return maybeBits(i->ops[0], depth - 1) | maybeBits(i->ops[1], depth - 1);
        if (i->sub == LShr && constInt(i->ops[1], k) && k < w) return maybeBits(i->ops[0], depth - 1) >> k;
        if (i->sub == Shl && constInt(i->ops[1], k) && k < w) return (maybeBits(i->ops[0], depth - 1) << k) & all;
        return all;
      case Op::Cast:
        if (i->sub == ZExt && isScalarInt(i->ops[0]->type)) return widthMask(i->ops[0]->type->n);
        return all;
      case Op::Cmp:
        return isScalarInt(i->type) ? 1 : all;
      case Op::Select:
        return maybeBits(i->ops[1], depth - 1) | maybeBits(i->ops[2], depth - 1);
      default:
        return all;
    }
  }

  Value* foldBinOp(Instruction* i) {
    Type* t = i->type;
    const unsigned w = t->n;
    const uint64_t mask = widthMask(w);
    Value *a = i->ops[0], *b = i->ops[1];
    uint64_t x = 0, y = 0;
    const bool ca = constInt(a, x), cb = constInt(b, y);
    if (ca && cb) {
      const int64_t sx = signExtend(x, w), sy = signExtend(y, w);
      const bool overflow = sx == signExtend(1ull << (w - 1), w) && sy == -1;  // INT_MIN / -1
      switch (i->sub) {
        case Add: return cint(t, x + y);
        case Sub: return cint(t, x - y);
        case Mul: return cint(t, x * y);
        case And: return cint(t, x & y);
        case Or: return cint(t, x | y);
        case Xor: return cint(t, x ^ y);
        case Shl: return y < w ? cint(t, x << y) : nullptr;
        case LShr: return y < w ? cint(t, x >> y) : nullptr;
        case AShr: return y < w ? cint(t, uint64_t(sx >> y)) : nullptr;
        case UDiv: return y ? cint(t, x / y) : nullptr;
        case URem: return y ? cint(t, x % y) : nullptr;
        case SDiv: return y && !overflow ? cint(t, uint64_t(sx / sy)) : nullptr;
        case SRem: return y && !overflow ? cint(t, uint64_t(sx % sy)) : nullptr;
        default: return nullptr;
      }
    }
    switch (i->sub) {
      case Add:
        if (cb && y == 0) return a;
        if (ca && x == 0) return b;
        break;
      case Sub:
        if (cb && y == 0) return a;
        if (a == b) return cint(t, 0);
        break;
      case Mul:
        if (cb && y == 1) return a;
        if (ca && x == 1) return b;
        if ((cb && y == 0) || (ca && x == 0)) return cint(t, 0);
        break;
      case And:
        if (cb && y == mask) return a;
        if (ca && x == mask) return b;
        if ((cb && y == 0) || (ca && x == 0)) return cint(t, 0);
        if (a == b) return a;
        if (cb && (maybeBits(a) & ~y & mask) == 0) return a;
        if (ca && (maybeBits(b) & ~x & mask) == 0) return b;
        if (cb && (maybeBits(a) & y) == 0) return cint(t, 0);
        if (ca && (maybeBits(b) & x) == 0) return cint(t, 0);
        break;
      case Or:
        if (cb && y == 0) return a;
        if (ca && x == 0) return b;
        if ((cb && y == mask) || (ca && x == mask)) return cint(t, mask);
        if (a == b) return a;
        break;
      case Xor:
        if (cb && y == 0) return a;
        if (ca && x == 0) return b;
        if (a == b) return cint(t, 0);
        break;
      case Shl:
      case LShr:
      case AShr:
        if (cb && y == 0) return a;
        if (ca && x == 0) return cint(t, 0);
        if (i->sub == LShr && cb && y < w && (maybeBits(a) >> y) == 0) return cint(t, 0);
        break;
      case UDiv:
      case SDiv:
        if (cb && y == 1) return a;
        break;
      case URem:
        if (cb && y == 1) return cint(t, 0);
        break;
      default: break;
    }
    return nullptr;
  }

  Value* foldCast(Instruction* i) {
    Value* a = i->ops[0];
    Type *from = a->type, *to = i->type;
    uint64_t x;
    if (i->sub == BitCast) {
      if (from == to) return a;
      if (a->vk == Value::Inst) {
        const auto* p = static_cast<const Instruction*>(a);
        if (p->op == Op::Cast && p->sub == BitCast && p->ops[0]->type == to) return p->ops[0];
      }
      if (a->vk == Value::Const) {
        const auto* c = static_cast<const Constant*>(a);
        if (isScalarInt(from) && isFloat(to) && constInt(a, x)) return m_.constFPBits(to, x);
        if (isFloat(from) && isScalarInt(to) && (c->ck == Constant::FP || c->ck == Constant::Null))
          return cint(to, c->ck == Constant::FP ? c->bits : 0);
      }
      return nullptr;
    }
    if (!isScalarInt(from) || !isScalarInt(to)) return nullptr;
    if (constInt(a, x)) {
      switch (i->sub) {
        case Trunc:
        case ZExt: return cint(to, x);
        case SExt: return cint(to, uint64_t(signExtend(x, from->n)));
        default: return nullptr;
      }
    }
    if (i->sub == Trunc && a->vk == Value::Inst) {
      const auto* p = static_cast<const Instruction*>(a);
      if (p->op == Op::Cast && (p->sub == ZExt || p->sub == SExt) && p->ops[0]->type == to) return p->ops[0];
    }
    return nullptr;
  }

  Value* foldCmp(Instruction* i) {
    Value *a = i->ops[0], *b = i->ops[1];
    if (i->sub < 32 || !isScalarInt(a->type) || !isScalarInt(i->type)) return nullptr;  // fcmp, vectors
    const unsigned w = a->type->n;
    const uint64_t mask = widthMask(w);
    uint64_t x = 0, y = 0;
    const bool ca = constInt(a, x), cb = constInt(b, y);
    auto result = [&](bool r) { return cint(i->type, r); };
    enum { EQ = 32, NE, UGT, UGE, ULT, ULE, SGT, SGE, SLT, SLE };
    if (ca && cb) {
      const int64_t sx = signExtend(x, w), sy = signExtend(y, w);
      switch (i->sub) {
        case EQ: return result(x == y);
        case NE: return result(x != y);
        case UGT: return result(x > y);
        case UGE: return result(x >= y);
        case ULT: return result(x < y);
        case ULE: return result(x <= y);
        case SGT: return result(sx > sy);
        case SGE: return result(sx >= sy);
        case SLT: return result(sx < sy);
        case SLE: return result(sx <= sy);
        default: return nullptr;
      }
    }
    if (a == b) {
      switch (i->sub) {
        case EQ: case UGE: case ULE: case SGE: case SLE: return result(true);
        case NE: case UGT: case ULT: case SGT: case SLT: return result(false);
        default: return nullptr;
      }
    }
    if (cb) {
      if (i->sub == ULT && y == 0) return result(false);
      if (i->sub == UGE && y == 0) return result(true);
      if (i->sub == UGT && y == mask) return result(false);
      if (i->sub == ULE && y == mask) return result(true);
      if (w == 1 && i->sub == EQ && y == 1) return a;
      if (w == 1 && i->sub == NE && y == 0) return a;
      // A value whose possible bits are all below the constant is below it.
      const uint64_t m = maybeBits(a);
      if (m < y && (i->sub == ULT || i->sub == ULE || i->sub == NE)) return result(true);
      if (m < y && (i->sub == UGT || i->sub == UGE || i->sub == EQ)) return result(false);
    }
    return nullptr;
  }

  Value* foldSelect(Instruction* i) {
    Value *c = i->ops[0], *a = i->ops[1], *b = i->ops[2];
    uint64_t k, x, y;
    if (constInt(c, k)) return (k & 1) ? a : b;
    if (a == b) return a;
    if (isScalarInt(i->type) && i->type->n == 1 && constInt(a, x) && constInt(b, y) && x == 1 && y == 0) return c;
    return nullptr;
  }

  Value* foldFNeg(Instruction* i) {
    Value* a = i->ops[0];
    if (a->vk == Value::Inst) {
      const auto* p = static_cast<const Instruction*>(a);
      if (p->op == Op::FNeg) return p->ops[0];
    }
    if (a->vk == Value::Const && isFloat(i->type)) {
      const auto* c = static_cast<const Constant*>(a);
      if (c->ck != Constant::FP && c->ck != Constant::Null) return nullptr;
      unsigned w = i->type->kind == Type::Half ? 16 : i->type->kind == Type::Float ? 32 : 64;
      return m_.constFPBits(i->type, (c->ck == Constant::FP ? c->bits : 0) ^ (1ull << (w - 1)));
    }
    return nullptr;
  }

  // extractvalue through insertvalues at other fields, and out of constant aggregates (single-level indices).
  Value* foldExtractValue(Instruction* i) {
    if (i->idx.size() != 1) return nullptr;
    Value* v = i->ops[0];
    bool moved = false;
    while (v->vk == Value::Inst) {
      const auto* p = static_cast<const Instruction*>(v);
      if (p->op != Op::InsertValue || p->idx.size() != 1) break;
      if (p->idx[0] == i->idx[0]) return p->ops[1];
      v = p->ops[0];
      moved = true;
    }
    if (v->vk == Value::Const) {
      const auto* c = static_cast<const Constant*>(v);
      if (c->ck == Constant::Aggregate && i->idx[0] < c->ops.size()) return c->ops[i->idx[0]];
      if (c->ck == Constant::Undef) return m_.constUndef(i->type);
    }
    if (moved) i->ops[0] = v;  // the same field, read from before the inserts at other fields
    return nullptr;
  }

  Value* simplify(Instruction* i) {
    switch (i->op) {
      case Op::BinOp: return isScalarInt(i->type) ? foldBinOp(i) : nullptr;
      case Op::Cast: return foldCast(i);
      case Op::Cmp: return foldCmp(i);
      case Op::Select: return foldSelect(i);
      case Op::FNeg: return foldFNeg(i);
      case Op::ExtractValue: return foldExtractValue(i);
      default: return nullptr;
    }
  }

  Module& m_;
  Function* f_;
  uint32_t ibase_ = 0;
  unsigned n_ = 0;
  std::vector<Value*> repl_;
};

}  // namespace

bool simplifyInstructions(Module& m, Function* f) {
  if (f->blocks.empty()) return false;
  return Simplifier(m, f).run();
}

}  // namespace spirv2air
