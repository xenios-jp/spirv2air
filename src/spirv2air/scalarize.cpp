// Exact scalarizer. Walks the blocks in reverse post-order (definitions before uses, phis aside) and gives each
// elementwise vector instruction one scalar instruction per lane; a lane of a value that is not scalarized is an
// extractelement placed right after that value's definition, made once per (value, lane). Insertelement and
// shufflevector with constant indices only move lanes around, so they create no instruction. Scalar phis are created
// empty and filled at the end, when every block has been seen. A scalarized vector some remaining instruction still
// uses (a store, a call, a return, an extractelement with a variable index, ...) is rebuilt with an insertelement chain
// right after its lanes; an extractelement with a constant index just takes the lane. Nothing is evaluated or
// reordered across instructions, so every lane computes what it computed inside the vector.
#include "spirv2air/scalarize.h"

#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

bool constantInt(const Value* v, uint64_t& out) {
  if (v->vk != Value::Const) return false;
  const auto* c = static_cast<const Constant*>(v);
  if (c->ck != Constant::Int && c->ck != Constant::Null) return false;
  out = c->ck == Constant::Int ? c->bits : 0;
  return true;
}

bool isVector(const Value* v) { return v->type->kind == Type::Vector; }

bool isUndef(const Value* v) { return v->vk == Value::Const && static_cast<const Constant*>(v)->ck == Constant::Undef; }

class Scalarizer {
 public:
  Scalarizer(Module& m, Function* f) : m_(m), f_(f) {
    // Instructions that get state are numbered densely as they are met (tag - ibase_), so that state is found by index,
    // not by hashing; a function without vectors numbers nothing.
    for (const BasicBlock* b : f->blocks) total_ += unsigned(b->insts.size());
    ibase_ = m.allocTags(total_);
    bbase_ = m.allocTags(unsigned(f->blocks.size()));
    for (size_t k = 0; k < f->blocks.size(); ++k) f->blocks[k]->tag = bbase_ + uint32_t(k);
  }

  void run() {
    for (BasicBlock* b : reversePostOrder())
      for (Instruction* i : b->insts) visit(i);
    if (order_.empty()) return;
    for (Instruction* phi : phis_)
      for (size_t j = 0; j < phi->ops.size(); ++j)
        for (unsigned k = 0; k < phi->type->n; ++k) {
          auto* p = static_cast<Instruction*>(lanesOf(phi)[k]);
          p->ops.push_back(lane(phi->ops[j], k));
          p->bbs.push_back(phi->bbs[j]);
        }
    rebuildVectors();
    for (BasicBlock* b : f_->blocks) rewriteBlock(b);
  }

 private:
  Module& m_;
  Function* f_;
  // Per original instruction, created on first use.
  struct Info {
    std::vector<Value*> lanes;                     // scalarized value -> its lane values
    std::vector<Value*> extracted;                 // other vector -> its extractelements
    std::vector<Instruction*> body, tail, extras;  // lane instructions, rebuilt-vector chain, extracts placed after it
  };
  std::deque<Info> infos_;
  std::vector<uint32_t> info_;  // by instruction id: 1 + index into infos_, or 0
  uint32_t ibase_ = 0, bbase_ = 0;
  unsigned n_ = 0, total_ = 0;  // instructions numbered so far, and instructions in the function
  std::unordered_map<const Value*, std::vector<Value*>> argExtracted_;  // arguments and other non-instruction vectors
  std::vector<Instruction*> argExtras_, order_, phis_;
  std::vector<char> removed_;  // by instruction id
  std::vector<Value*> repl_;   // by instruction id: its replacement, or null (only original instructions are replaced)

  // The id of an original instruction of this function, or ~0u (new instructions, constants, arguments).
  unsigned id(const Value* v) const {
    unsigned k = v->tag - ibase_;
    return v->vk == Value::Inst && k < n_ ? k : ~0u;
  }
  Info* find(const Value* v) {
    unsigned k = id(v);
    return k == ~0u || !info_[k] ? nullptr : &infos_[info_[k] - 1];
  }
  // The id of an original instruction, assigning one on first use.
  unsigned number(Instruction* i) {
    if (id(i) == ~0u) {
      if (info_.empty()) info_.assign(total_, 0), removed_.assign(total_, 0), repl_.assign(total_, nullptr);
      i->tag = ibase_ + n_++;
    }
    return i->tag - ibase_;
  }
  Info& get(const Value* v) {
    uint32_t& slot = info_[number(const_cast<Instruction*>(static_cast<const Instruction*>(v)))];
    if (!slot) infos_.emplace_back(), slot = uint32_t(infos_.size());
    return infos_[slot - 1];
  }
  bool scalarized(const Value* v) {
    const Info* in = find(v);
    return in && !in->lanes.empty();
  }
  std::vector<Value*>& lanesOf(const Value* v) { return find(v)->lanes; }
  Value* resolveRepl(Value* v) {
    for (unsigned k = id(v); k != ~0u && repl_[k]; k = id(v)) v = repl_[k];
    return v;
  }

  std::vector<BasicBlock*> reversePostOrder() const {
    std::vector<BasicBlock*> post;
    std::vector<char> seen(f_->blocks.size(), 0);
    seen[0] = 1;
    std::vector<std::pair<BasicBlock*, size_t>> stack{{f_->blocks[0], 0}};
    while (!stack.empty()) {
      auto& [b, k] = stack.back();
      const Instruction* t = b->insts.empty() ? nullptr : b->insts.back();
      size_t n = t && (t->op == Op::Br || t->op == Op::Switch) ? t->bbs.size() : 0;
      if (k < n) {
        BasicBlock* s = t->bbs[k++];
        if (!seen[s->tag - bbase_]) seen[s->tag - bbase_] = 1, stack.push_back({s, 0});
      } else {
        post.push_back(b);
        stack.pop_back();
      }
    }
    return {post.rbegin(), post.rend()};
  }

  // Lane k of any vector value.
  Value* lane(Value* v, unsigned k) {
    Type* elem = v->type->elem;
    if (const Info* in = find(v); in && !in->lanes.empty()) return in->lanes[k];
    if (v->vk == Value::Const) {
      const auto* c = static_cast<const Constant*>(v);
      if (c->ck == Constant::Aggregate) return c->ops[k];
      if (c->ck == Constant::Null) return m_.constNull(elem);
      if (c->ck == Constant::Undef) return m_.constUndef(elem);
      if (c->ck == Constant::Poison) return m_.constPoison(elem);
    }
    std::vector<Value*>& made = v->vk == Value::Inst ? get(v).extracted : argExtracted_[v];
    if (made.empty()) made.assign(v->type->n, nullptr);
    if (!made[k]) {
      Instruction* x = m_.newInst(Op::ExtractElement, elem);
      x->ops = {v, m_.constInt(m_.intTy(32), k)};
      (v->vk == Value::Inst ? get(v).extras : argExtras_).push_back(x);
      made[k] = x;
    }
    return made[k];
  }

  void visit(Instruction* i) {
    if (i->op == Op::ExtractElement) {
      uint64_t k;
      const Info* in = find(i->ops[0]);
      if (!in || in->lanes.empty() || !constantInt(i->ops[1], k) || k >= in->lanes.size()) return;
      Value* r = resolveRepl(in->lanes[k]);
      repl_[number(i)] = r;
      removed_[id(i)] = 1;
      return;
    }
    if (!isVector(i)) return;
    const unsigned n = i->type->n;
    Type* elem = i->type->elem;
    std::vector<Value*> out(n);
    switch (i->op) {
      case Op::Cast:
        if (!isVector(i->ops[0]) || i->ops[0]->type->n != n) return;
        [[fallthrough]];
      case Op::BinOp:
      case Op::FNeg:
      case Op::Cmp:
      case Op::Select:
      case Op::Phi:
        for (unsigned k = 0; k < n; ++k) {
          Instruction* s = m_.newInst(i->op, elem);
          s->sub = i->sub;
          s->flags = i->flags;
          if (i->op == Op::Select) {
            s->ops = {isVector(i->ops[0]) ? lane(i->ops[0], k) : i->ops[0], lane(i->ops[1], k), lane(i->ops[2], k)};
          } else if (i->op != Op::Phi) {
            for (Value* o : i->ops) s->ops.push_back(lane(o, k));
          }
          get(i).body.push_back(s);
          out[k] = s;
        }
        if (i->op == Op::Phi) phis_.push_back(i);  // incoming lanes are filled in once every block has been seen
        break;
      case Op::InsertElement: {
        uint64_t k;
        if (!constantInt(i->ops[2], k) || k >= n) return;
        for (unsigned j = 0; j < n; ++j) out[j] = j == k ? i->ops[1] : lane(i->ops[0], j);
        break;
      }
      case Op::ShuffleVector: {
        const auto* mask = static_cast<const Constant*>(i->ops[2]);
        const unsigned na = i->ops[0]->type->n;
        for (unsigned j = 0; j < n; ++j) {
          const Value* e = mask->ck == Constant::Aggregate ? mask->ops[j] : mask;
          uint64_t src;
          if (mask->ck == Constant::Undef || mask->ck == Constant::Poison || isUndef(e)) {
            out[j] = m_.constUndef(elem);
          } else if (constantInt(e, src)) {
            out[j] = src < na ? lane(i->ops[0], unsigned(src)) : lane(i->ops[1], unsigned(src - na));
          } else {
            return;
          }
        }
        break;
      }
      default:
        return;
    }
    get(i).lanes = std::move(out);
    removed_[id(i)] = 1;
    order_.push_back(i);
  }

  // Gives each scalarized vector something still uses an insertelement chain over its lanes (undef lanes skipped).
  void rebuildVectors() {
    std::vector<char> used(n_, 0);
    for (BasicBlock* b : f_->blocks)
      for (Instruction* i : b->insts)
        if (id(i) == ~0u || !removed_[id(i)])
          for (Value* v : i->ops)
            if (scalarized(v)) used[id(v)] = 1;
    for (Instruction* i : order_) {
      if (!used[id(i)]) continue;
      Value* cur = m_.constUndef(i->type);
      for (unsigned k = 0; k < i->type->n; ++k) {
        Value* l = lanesOf(i)[k];
        if (isUndef(l)) continue;
        Instruction* ins = m_.newInst(Op::InsertElement, i->type);
        ins->ops = {cur, l, m_.constInt(m_.intTy(32), k)};
        get(i).tail.push_back(ins);
        cur = ins;
      }
      repl_[id(i)] = cur;
    }
  }

  // The block with scalarized instructions replaced by their lanes, extracts after their definitions and rebuilt
  // vectors after their lanes (all of them after the block's phis), and every operand resolved through repl_.
  void rewriteBlock(BasicBlock* b) {
    std::vector<Instruction*> phiPart, postPhi, rest;
    auto append = [](std::vector<Instruction*>& to, const std::vector<Instruction*>& from) {
      to.insert(to.end(), from.begin(), from.end());
    };
    for (Instruction* i : b->insts) {
      const bool phi = i->op == Op::Phi, lanes = scalarized(i);
      if (id(i) != ~0u && removed_[id(i)] && !lanes) continue;
      if (lanes) {
        const Info& in = *find(i);
        append(phi ? phiPart : rest, in.body);
        append(phi ? postPhi : rest, in.tail);
      } else {
        (phi ? phiPart : rest).push_back(i);
        if (const Info* in = find(i)) append(phi ? postPhi : rest, in->extras);
      }
    }
    std::vector<Instruction*>& list = b->insts;
    list = std::move(phiPart);
    list.insert(list.end(), postPhi.begin(), postPhi.end());
    if (b == f_->blocks[0]) list.insert(list.end(), argExtras_.begin(), argExtras_.end());
    list.insert(list.end(), rest.begin(), rest.end());
    for (Instruction* i : list)
      for (Value*& v : i->ops)
        if (unsigned k = id(v); k != ~0u && repl_[k]) v = repl_[k];
  }
};

}  // namespace

void scalarizeFunction(Module& m, Function* f) {
  if (f->blocks.empty()) return;
  Scalarizer(m, f).run();
}

}  // namespace spirv2air
