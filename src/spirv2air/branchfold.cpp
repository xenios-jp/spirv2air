// See branchfold.h. The evaluation rules are the ones instsimplify.cpp and cleanup.cpp apply to the same instructions
// (same widths, same exclusions); a condition is only replaced when every value on its way is a known integer, so the
// replacement is a value the later rounds compute as well.
#include "spirv2air/branchfold.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

uint64_t widthMask(unsigned w) { return w >= 64 ? ~0ull : (1ull << w) - 1; }

int64_t signExtend(uint64_t v, unsigned w) { return w >= 64 ? int64_t(v) : int64_t(v << (64 - w)) >> (64 - w); }

bool isScalarInt(const Type* t) { return t->kind == Type::Int; }

// A scalar integer constant (a null one is zero).
bool constInt(const Value* v, uint64_t& out) {
  if (v->vk != Value::Const || !isScalarInt(v->type)) return false;
  const auto* c = static_cast<const Constant*>(v);
  if (c->ck != Constant::Int && c->ck != Constant::Null) return false;
  out = c->ck == Constant::Int ? c->bits : 0;
  return true;
}

const SmallVec<BasicBlock*, 2>& successors(const BasicBlock* b) {
  static const SmallVec<BasicBlock*, 2> none{};
  if (b->insts.empty()) return none;
  const Instruction* t = b->insts.back();
  return t->op == Op::Br || t->op == Op::Switch ? t->bbs : none;
}

class BranchFolder {
 public:
  BranchFolder(Module& m, Function* f) : m_(m), f_(f) {
    unsigned n = 0;
    for (const BasicBlock* b : f->blocks) n += unsigned(b->insts.size());
    ibase_ = m.allocTags(n);
    bbase_ = m.allocTags(unsigned(f->blocks.size()));
    state_.assign(n, Unseen);
    value_.assign(n, 0);
    home_.assign(n, 0);
    for (size_t b = 0; b < f->blocks.size(); ++b) {
      f->blocks[b]->tag = bbase_ + uint32_t(b);
      for (Instruction* i : f->blocks[b]->insts) {
        i->tag = ibase_ + n_;
        home_[n_++] = uint32_t(b);
      }
    }
  }

  bool run() {
    const size_t nb = f_->blocks.size();
    if (nb == 0) return false;
    reversePostOrder();
    block_.assign(nb, Pending);
    only_.assign(nb, nullptr);
    std::vector<char> live(nb, 0);
    live[0] = 1;
    std::vector<std::pair<Instruction*, uint64_t>> folds;
    for (int b : rpo_) {
      if (!live[b]) {
        block_[b] = Dead;
        continue;
      }
      BasicBlock* blk = f_->blocks[b];
      Instruction* t = blk->insts.back();
      uint64_t c = 0;
      if (t->op == Op::Br && t->bbs.size() == 2 && t->bbs[0] != t->bbs[1] && eval(t->ops[0], c, 0)) {
        only_[b] = t->bbs[(c & 1) ? 0 : 1];
        if (t->ops[0]->vk != Value::Const) folds.emplace_back(t, c & 1);
      } else if (t->op == Op::Switch && eval(t->ops[0], c, 0)) {
        BasicBlock* target = t->bbs[0];
        for (size_t k = 1; k < t->bbs.size(); ++k) {
          uint64_t cv = 0;
          if (constInt(t->ops[k], cv) && cv == c) {
            target = t->bbs[k];
            break;
          }
        }
        only_[b] = target;
        if (t->ops[0]->vk != Value::Const) folds.emplace_back(t, c);
      }
      block_[b] = Live;
      for (const BasicBlock* s : successors(blk))
        if (!only_[b] || only_[b] == s) live[s->tag - bbase_] = 1;
    }
    for (auto& [t, c] : folds) t->ops[0] = m_.constInt(t->ops[0]->type, int64_t(c));
    return !folds.empty();
  }

 private:
  enum : uint8_t { Unseen, Active, Known, Unknown };
  enum : uint8_t { Pending, Live, Dead };

  unsigned id(const Value* v) const {
    unsigned k = v->tag - ibase_;
    return v->vk == Value::Inst && k < n_ ? k : ~0u;
  }

  // Block numbers in reverse post-order from the entry over every edge.
  void reversePostOrder() {
    const size_t nb = f_->blocks.size();
    std::vector<char> seen(nb, 0);
    std::vector<std::pair<int, size_t>> stack{{0, 0}};
    seen[0] = 1;
    std::vector<int> post;
    post.reserve(nb);
    while (!stack.empty()) {
      auto& [b, k] = stack.back();
      const auto& succ = successors(f_->blocks[b]);
      if (k < succ.size()) {
        const int s = int(succ[k++]->tag - bbase_);
        if (!seen[s]) seen[s] = 1, stack.push_back({s, 0});
      } else {
        post.push_back(b);
        stack.pop_back();
      }
    }
    rpo_.assign(post.rbegin(), post.rend());
    reachable_.assign(nb, 0);
    for (int b : rpo_) reachable_[b] = 1;
  }

  // Is the edge from `p` into block `to` one a branch can take, as far as the walk has got? 1: yes, 0: no, -1: not
  // known yet (`p` comes later in the order, so the edge is a back edge).
  int edgeTaken(const BasicBlock* p, const BasicBlock* to) const {
    const unsigned pb = p->tag - bbase_;
    if (pb >= block_.size() || !reachable_[pb]) return 0;
    if (block_[pb] == Dead) return 0;
    if (block_[pb] == Pending) return -1;
    return !only_[pb] || only_[pb] == to;
  }

  // The integer the value holds on every execution, if it is built from constants alone.
  bool eval(const Value* v, uint64_t& out, int depth) {
    if (v->vk == Value::Const) return constInt(v, out);
    const unsigned k = id(v);
    if (k == ~0u) return false;
    if (state_[k] == Known) return out = value_[k], true;
    if (state_[k] != Unseen || depth > 48) return false;
    state_[k] = Active;
    uint64_t r = 0;
    const bool known = compute(static_cast<const Instruction*>(v), r, depth + 1);
    state_[k] = known ? Known : Unknown;
    if (known) value_[k] = r;
    out = r;
    return known;
  }

  bool compute(const Instruction* i, uint64_t& out, int depth) {
    switch (i->op) {
      case Op::BinOp: return isScalarInt(i->type) && binop(i, out, depth);
      case Op::Cast: return cast(i, out, depth);
      case Op::Cmp: return cmp(i, out, depth);
      case Op::Select: {
        uint64_t c = 0, x = 0, y = 0;
        if (eval(i->ops[0], c, depth)) return eval(i->ops[(c & 1) ? 1 : 2], out, depth);
        if (!eval(i->ops[1], x, depth) || !eval(i->ops[2], y, depth) || x != y) return false;
        return out = x, true;
      }
      case Op::Phi: return phi(i, out, depth);
      case Op::Call: return call(i, out, depth);
      case Op::ExtractValue: {
        if (i->idx.size() != 1) return false;
        const Value* v = i->ops[0];
        while (v->vk == Value::Inst) {
          const auto* p = static_cast<const Instruction*>(v);
          if (p->op != Op::InsertValue || p->idx.size() != 1) break;
          if (p->idx[0] == i->idx[0]) return eval(p->ops[1], out, depth);
          v = p->ops[0];
        }
        if (v->vk != Value::Const) return false;
        const auto* c = static_cast<const Constant*>(v);
        return c->ck == Constant::Aggregate && i->idx[0] < c->ops.size() && eval(c->ops[i->idx[0]], out, depth);
      }
      case Op::ExtractElement: return extractElement(i, out, depth);
      default: return false;
    }
  }

  bool binop(const Instruction* i, uint64_t& out, int depth) {
    const Type* t = i->type;
    const unsigned w = t->n;
    const uint64_t mask = widthMask(w);
    uint64_t x = 0, y = 0;
    const bool ca = eval(i->ops[0], x, depth), cb = eval(i->ops[1], y, depth);
    if (ca && cb) {
      const int64_t sx = signExtend(x, w), sy = signExtend(y, w);
      const bool overflow = sx == signExtend(1ull << (w - 1), w) && sy == -1;  // INT_MIN / -1
      uint64_t r;
      switch (i->sub) {
        case Add: r = x + y; break;
        case Sub: r = x - y; break;
        case Mul: r = x * y; break;
        case And: r = x & y; break;
        case Or: r = x | y; break;
        case Xor: r = x ^ y; break;
        case Shl:
          if (y >= w) return false;
          r = x << y;
          break;
        case LShr:
          if (y >= w) return false;
          r = x >> y;
          break;
        case AShr:
          if (y >= w) return false;
          r = uint64_t(sx >> y);
          break;
        case UDiv:
          if (!y) return false;
          r = x / y;
          break;
        case URem:
          if (!y) return false;
          r = x % y;
          break;
        case SDiv:
          if (!y || overflow) return false;
          r = uint64_t(sx / sy);
          break;
        case SRem:
          if (!y || overflow) return false;
          r = uint64_t(sx % sy);
          break;
        default: return false;
      }
      return out = r & mask, true;
    }
    // One operand known: the cases where it alone decides the result.
    switch (i->sub) {
      case And:
        if ((cb && y == 0) || (ca && x == 0)) return out = 0, true;
        break;
      case Or:
        if ((cb && y == mask) || (ca && x == mask)) return out = mask, true;
        break;
      case Mul:
        if ((cb && y == 0) || (ca && x == 0)) return out = 0, true;
        break;
      case Shl:
      case LShr:
      case AShr:
        if (ca && x == 0) return out = 0, true;
        break;
      default: break;
    }
    return false;
  }

  bool cast(const Instruction* i, uint64_t& out, int depth) {
    const Type *from = i->ops[0]->type, *to = i->type;
    if (!isScalarInt(from) || !isScalarInt(to)) return false;
    uint64_t x;
    if (!eval(i->ops[0], x, depth)) return false;
    switch (i->sub) {
      case Trunc:
      case ZExt: return out = x & widthMask(to->n), true;
      case SExt: return out = uint64_t(signExtend(x, from->n)) & widthMask(to->n), true;
      default: return false;
    }
  }

  bool cmp(const Instruction* i, uint64_t& out, int depth) {
    const Value *a = i->ops[0], *b = i->ops[1];
    if (i->sub < 32 || !isScalarInt(a->type) || !isScalarInt(i->type)) return false;  // fcmp, vectors
    enum { EQ = 32, NE, UGT, UGE, ULT, ULE, SGT, SGE, SLT, SLE };
    const unsigned w = a->type->n;
    uint64_t x = 0, y = 0;
    const bool ca = eval(a, x, depth), cb = eval(b, y, depth);
    if (ca && cb) {
      const int64_t sx = signExtend(x, w), sy = signExtend(y, w);
      switch (i->sub) {
        case EQ: return out = x == y, true;
        case NE: return out = x != y, true;
        case UGT: return out = x > y, true;
        case UGE: return out = x >= y, true;
        case ULT: return out = x < y, true;
        case ULE: return out = x <= y, true;
        case SGT: return out = sx > sy, true;
        case SGE: return out = sx >= sy, true;
        case SLT: return out = sx < sy, true;
        case SLE: return out = sx <= sy, true;
        default: return false;
      }
    }
    if (cb) {
      const uint64_t mask = widthMask(w);
      if (i->sub == ULT && y == 0) return out = 0, true;
      if (i->sub == UGE && y == 0) return out = 1, true;
      if (i->sub == UGT && y == mask) return out = 0, true;
      if (i->sub == ULE && y == mask) return out = 1, true;
    }
    return false;
  }

  bool phi(const Instruction* i, uint64_t& out, int depth) {
    const BasicBlock* home = f_->blocks[home_[id(i)]];
    bool any = false;
    uint64_t first = 0;
    for (size_t e = 0; e < i->ops.size(); ++e) {
      const int taken = edgeTaken(i->bbs[e], home);
      if (taken < 0) return false;
      if (!taken) continue;
      uint64_t x;
      if (!eval(i->ops[e], x, depth)) return false;
      if (any && x != first) return false;
      first = x;
      any = true;
    }
    return any && (out = first, true);
  }

  bool call(const Instruction* i, uint64_t& out, int depth) {
    if (i->ops.empty() || i->ops[0]->vk != Value::Func || i->ops.size() != 4) return false;
    const Value* callee = i->ops[0];
    auto [it, fresh] = extractBits_.try_emplace(callee, 0);
    if (fresh) {
      if (callee->name == "air.extract_bits.u.i32") it->second = 1;
      if (callee->name == "air.extract_bits.s.i32") it->second = 2;
    }
    if (!it->second || !isScalarInt(i->type) || i->type->n != 32) return false;
    uint64_t x, y, z;
    if (!eval(i->ops[1], x, depth) || !eval(i->ops[2], y, depth) || !eval(i->ops[3], z, depth)) return false;
    if (z == 0 || y >= 32 || z > 32 - y) return false;
    x = (x >> y) & widthMask(unsigned(z));
    return out = (it->second == 2 ? uint64_t(signExtend(x, unsigned(z))) & widthMask(32) : x), true;
  }

  // A lane of a vector: walks back through inserts at other constant lanes and shuffles to the value that supplies it.
  bool extractElement(const Instruction* x, uint64_t& out, int depth) {
    uint64_t k;
    if (!constInt(x->ops[1], k)) return false;
    const Value* v = x->ops[0];
    for (;;) {
      if (k >= v->type->n) return false;
      if (v->vk == Value::Const) {
        const auto* c = static_cast<const Constant*>(v);
        if (c->ck == Constant::Null) return out = 0, true;
        return c->ck == Constant::Aggregate && eval(c->ops[k], out, depth);
      }
      if (v->vk != Value::Inst) return false;
      const auto* i = static_cast<const Instruction*>(v);
      uint64_t j;
      if (i->op == Op::InsertElement && constInt(i->ops[2], j) && j < v->type->n) {
        if (j == k) return eval(i->ops[1], out, depth);
        v = i->ops[0];
      } else if (i->op == Op::ShuffleVector) {
        const auto* mask = static_cast<const Constant*>(i->ops[2]);
        const Value* lane = mask->ck == Constant::Aggregate ? mask->ops[k] : mask;
        uint64_t m;
        if (!constInt(lane, m)) return false;
        const unsigned na = i->ops[0]->type->n;
        v = m < na ? i->ops[0] : i->ops[1];
        k = m < na ? m : m - na;
      } else {
        return false;
      }
    }
  }

  Module& m_;
  Function* f_;
  uint32_t ibase_ = 0, bbase_ = 0;
  unsigned n_ = 0;
  std::vector<uint8_t> state_;
  std::vector<uint64_t> value_;
  std::vector<uint32_t> home_;  // the block each instruction is in
  std::vector<int> rpo_;
  std::vector<char> reachable_;
  std::vector<uint8_t> block_;
  std::vector<BasicBlock*> only_;  // the one successor a known condition takes
  std::unordered_map<const Value*, int> extractBits_;
};

}  // namespace

bool foldKnownBranches(Module& m, Function* f) { return BranchFolder(m, f).run(); }

}  // namespace spirv2air
