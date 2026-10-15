// D1 cleanup. Six transformations, repeated until none applies (folding a branch can orphan blocks, merging
// resolves phis, resolved phis can make a condition constant, removing a dead instruction can free its operands):
//   A  constant or same-target br / switch -> unconditional br, phi entries of the dropped edges removed
//   B  blocks unreachable from the entry are removed, with their entries in the phis of surviving blocks
//   C  a block whose only predecessor ends in `br` to it is appended to that predecessor
//   D  phis with one distinct incoming value (self-references ignored) are replaced by it
//   E  extractelement at a constant index from insertelement, shufflevector or a constant vector -> the element it
//      reads (an inserted scalar, a lane of a shuffle source, a constant), so vectors built only to be read lane by
//      lane become dead
//   F  mark and sweep of side-effect-free instructions no live instruction uses (this also removes dead phi cycles)
// Nothing is evaluated: E forwards values, and otherwise only instructions whose result is unused disappear, so every
// remaining value is computed as before. All walks are iterative; the largest corpus shader has over 6,000 blocks.
#include "spirv2air/cleanup.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

bool isTerminator(const Instruction* i) {
  return i->op == Op::Ret || i->op == Op::Br || i->op == Op::Unreachable || i->op == Op::Switch;
}

// The successors of a block, one per edge (a switch with two cases to one block lists it twice).
const SmallVec<BasicBlock*, 2>& successors(const BasicBlock* b) {
  static const SmallVec<BasicBlock*, 2> none{};
  if (b->insts.empty()) return none;
  const Instruction* t = b->insts.back();
  return t->op == Op::Br || t->op == Op::Switch ? t->bbs : none;
}

// A constant integer's value, or false for anything else (undef and poison included: they are not folded).
bool constantInt(const Value* v, uint64_t& out) {
  if (v->vk != Value::Const) return false;
  const auto* c = static_cast<const Constant*>(v);
  if (c->ck != Constant::Int && c->ck != Constant::Null) return false;
  out = c->ck == Constant::Int ? c->bits : 0;
  return true;
}

bool hasFunctionAttr(const AttrList* l, const Attr& a) {
  if (!l) return false;
  for (const auto& [index, set] : l->sets)
    if (index == AttrList::FunctionIndex && std::find(set->begin(), set->end(), a) != set->end()) return true;
  return false;
}

// A call is pure when its callee is declared readnone (the AttrClass::Pure intrinsics): same arguments, same result,
// no memory read or written, no other effect. readonly texture and bit-field calls are left alone on purpose.
bool pureCall(const Instruction* i) {
  static const Attr readnone = Attr::get("readnone");
  if (i->ops.empty() || i->ops[0]->vk != Value::Func) return false;
  return hasFunctionAttr(static_cast<const Function*>(i->ops[0])->attrs, readnone) ||
         hasFunctionAttr(i->attrs, readnone);
}

// Instructions that may be deleted when nothing uses their result.
bool removable(const Instruction* i) {
  switch (i->op) {
    case Op::BinOp:
    case Op::Cast:
    case Op::GEP:
    case Op::ExtractValue:
    case Op::InsertValue:
    case Op::ExtractElement:
    case Op::InsertElement:
    case Op::ShuffleVector:
    case Op::Cmp:
    case Op::Select:
    case Op::FNeg:
    case Op::Phi:
      return true;
    case Op::Load:
      return !i->isVolatile;
    case Op::Call:
      return pureCall(i);
    default:
      return false;
  }
}

class Cleaner {
 public:
  Cleaner(Module& m, Function* f) : m_(m), f_(f) {
    // Instructions get dense ids (tag - ibase_) so the per-instruction sets and the replacement map are vectors.
    unsigned n = 0;
    for (const BasicBlock* b : f->blocks) n += unsigned(b->insts.size());
    ibase_ = m.allocTags(n);
    n_ = 0;
    for (BasicBlock* b : f->blocks)
      for (Instruction* i : b->insts) i->tag = ibase_ + n_++;
    repl_.assign(n_, nullptr);
  }

  CleanupStats run() {
    // Every transformation gets a turn each round, unless it already found nothing to do on the IR as it is now (a
    // transformation that changed nothing would change nothing again) or is the sweep, which one run completes.
    for (bool changed = true; changed;) {
      changed = turn(0, &Cleaner::foldBranches);
      changed |= turn(1, &Cleaner::removeUnreachable);
      changed |= turn(2, &Cleaner::mergeBlocks);
      changed |= turn(3, &Cleaner::simplifyPhis);
      changed |= turn(4, &Cleaner::forwardExtracts);
      changed |= turn(5, &Cleaner::sweepDead, /*idempotent=*/true);
    }
    return stats_;
  }

 private:
  // Runs transformation `which` unless it is up to date: `clean_[which]` is the number of changes made when it last
  // left nothing to do.
  bool turn(int which, bool (Cleaner::*stage)(), bool idempotent = false) {
    if (clean_[which] == changes_) return false;
    bool changed = (this->*stage)();
    if (changed) ++changes_;
    if (!changed || idempotent) clean_[which] = changes_;
    return changed;
  }

  // Drops one phi entry for an edge from `from` in the phis at the start of `to`.
  static void dropEdge(BasicBlock* from, BasicBlock* to) {
    for (Instruction* i : to->insts) {
      if (i->op != Op::Phi) break;
      auto it = std::find(i->bbs.begin(), i->bbs.end(), from);
      if (it == i->bbs.end()) continue;
      i->ops.erase(i->ops.begin() + (it - i->bbs.begin()));
      i->bbs.erase(it);
    }
  }

  // The dense id of an instruction of this function, or ~0u.
  unsigned id(const Value* v) const {
    unsigned k = v->tag - ibase_;
    return v->vk == Value::Inst && k < n_ ? k : ~0u;
  }

  Value* resolve(Value* v) const {
    for (unsigned k = id(v); k != ~0u && repl_[k]; k = id(v)) v = repl_[k];
    return v;
  }

  // Numbers the blocks 0..n-1 in layout order and returns their base tag.
  uint32_t numberBlocks() {
    uint32_t base = m_.allocTags(unsigned(f_->blocks.size()));
    for (size_t k = 0; k < f_->blocks.size(); ++k) f_->blocks[k]->tag = base + uint32_t(k);
    return base;
  }

  // Rewrites every operand through the replacement map, then forgets it.
  void applyReplacements() {
    if (!pending_) return;
    for (BasicBlock* b : f_->blocks)
      for (Instruction* i : b->insts)
        for (Value*& v : i->ops) v = resolve(v);
    std::fill(repl_.begin(), repl_.end(), nullptr);
    pending_ = false;
  }

  void replace(const Instruction* from, Value* to) {
    repl_[id(from)] = to;
    pending_ = true;
  }

  // A: br on a constant or with equal targets, switch on a constant.
  bool foldBranches() {
    bool changed = false;
    for (BasicBlock* b : f_->blocks) {
      Instruction* t = b->insts.back();
      uint64_t c = 0;
      size_t keep = 0;
      if (t->op == Op::Br && t->bbs.size() == 2) {
        if (constantInt(t->ops[0], c))
          keep = (c & 1) ? 0 : 1;
        else if (t->bbs[0] != t->bbs[1])
          continue;
      } else if (t->op == Op::Switch && constantInt(t->ops[0], c)) {
        // The default (bbs[0]) unless a case value matches; case k has value ops[k] and destination bbs[k].
        for (size_t k = 1; k < t->bbs.size(); ++k) {
          uint64_t cv = 0;
          if (constantInt(t->ops[k], cv) && cv == c) {
            keep = k;
            break;
          }
        }
      } else {
        continue;
      }
      // Every other edge disappears, including duplicate edges to the kept target.
      BasicBlock* target = t->bbs[keep];
      for (size_t k = 0; k < t->bbs.size(); ++k)
        if (k != keep) dropEdge(b, t->bbs[k]);
      t->op = Op::Br;
      t->bbs = {target};
      t->ops.clear();
      ++stats_.foldedBranches;
      changed = true;
    }
    return changed;
  }

  // B: blocks no path from the entry reaches.
  bool removeUnreachable() {
    const uint32_t base = numberBlocks();
    const size_t nb = f_->blocks.size();
    std::vector<char> seen(nb, 0);
    std::vector<const BasicBlock*> stack{f_->blocks[0]};
    seen[0] = 1;
    size_t reached = 1;
    while (!stack.empty()) {
      const BasicBlock* b = stack.back();
      stack.pop_back();
      for (const BasicBlock* s : successors(b))
        if (!seen[s->tag - base]) seen[s->tag - base] = 1, ++reached, stack.push_back(s);
    }
    if (reached == nb) return false;
    std::vector<BasicBlock*> kept;
    for (BasicBlock* b : f_->blocks) {
      if (seen[b->tag - base]) {
        kept.push_back(b);
        continue;
      }
      for (BasicBlock* s : successors(b))
        if (seen[s->tag - base]) dropEdge(b, s);
      ++stats_.unreachableBlocks;
    }
    f_->blocks = std::move(kept);
    return true;
  }

  // C: appends a block to its sole predecessor when that predecessor's only successor is the block.
  bool mergeBlocks() {
    const uint32_t base = numberBlocks();
    std::vector<unsigned> preds(f_->blocks.size(), 0);
    for (const BasicBlock* b : f_->blocks)
      for (const BasicBlock* s : successors(b)) ++preds[s->tag - base];
    BasicBlock* entry = f_->blocks[0];
    std::vector<char> merged(f_->blocks.size(), 0);
    bool anyMerged = false;
    for (BasicBlock* p : f_->blocks) {
      if (merged[p->tag - base]) continue;
      for (;;) {
        Instruction* t = p->insts.back();
        if (t->op != Op::Br || t->bbs.size() != 1) break;
        BasicBlock* s = t->bbs[0];
        if (s == p || s == entry || preds[s->tag - base] != 1) break;
        size_t firstNonPhi = 0;
        while (firstNonPhi < s->insts.size() && s->insts[firstNonPhi]->op == Op::Phi) ++firstNonPhi;
        bool single = true;
        for (size_t k = 0; k < firstNonPhi; ++k) single &= s->insts[k]->ops.size() == 1;
        if (!single) break;
        for (size_t k = 0; k < firstNonPhi; ++k) replace(s->insts[k], s->insts[k]->ops[0]);
        p->insts.pop_back();
        p->insts.insert(p->insts.end(), s->insts.begin() + long(firstNonPhi), s->insts.end());
        // The merged block's successors now see p where they saw s.
        for (BasicBlock* succ : successors(p))
          for (Instruction* i : succ->insts) {
            if (i->op != Op::Phi) break;
            for (BasicBlock*& e : i->bbs)
              if (e == s) e = p;
          }
        merged[s->tag - base] = 1;
        anyMerged = true;
        ++stats_.mergedBlocks;
      }
    }
    if (!anyMerged) return false;
    f_->blocks.erase(
        std::remove_if(f_->blocks.begin(), f_->blocks.end(), [&](BasicBlock* b) { return merged[b->tag - base] != 0; }),
        f_->blocks.end());
    applyReplacements();
    return true;
  }

  // D: phis whose incoming values are all one value besides the phi itself.
  bool simplifyPhis() {
    bool any = false;
    std::vector<char> gone(n_, 0);
    for (bool changed = true; changed;) {
      changed = false;
      for (BasicBlock* b : f_->blocks)
        for (Instruction* phi : b->insts) {
          if (phi->op != Op::Phi) break;
          if (gone[id(phi)]) continue;
          Value* same = nullptr;
          bool trivial = true;
          for (Value*& v : phi->ops) {
            v = resolve(v);
            if (v == phi || v == same) continue;
            if (same) {
              trivial = false;
              break;
            }
            same = v;
          }
          if (!trivial) continue;
          replace(phi, same ? same : m_.constUndef(phi->type));
          gone[id(phi)] = 1;
          changed = any = true;
          ++stats_.trivialPhis;
        }
    }
    if (!any) return false;
    for (BasicBlock* b : f_->blocks)
      b->insts.erase(std::remove_if(b->insts.begin(), b->insts.end(), [&](Instruction* i) { return gone[id(i)] != 0; }),
                     b->insts.end());
    applyReplacements();
    return true;
  }

  // E: an extract walks back through inserts at other constant indices and through shuffles (rewriting itself to
  // read the shuffle's source) until it reaches the lane's value. Out-of-range indices are left alone (poison).
  bool forwardExtracts() {
    bool any = false;
    for (BasicBlock* b : f_->blocks)
      for (Instruction* x : b->insts) {
        if (x->op != Op::ExtractElement) continue;
        uint64_t k;
        if (!constantInt(x->ops[1], k)) continue;
        Value* found = nullptr;
        for (Value* v = x->ops[0]; !found;) {
          if (k >= v->type->n) break;
          if (v->vk == Value::Const) {
            const auto* c = static_cast<const Constant*>(v);
            if (c->ck == Constant::Aggregate) found = c->ops[k];
            if (c->ck == Constant::Null) found = m_.constNull(x->type);
            if (c->ck == Constant::Undef) found = m_.constUndef(x->type);
            break;
          }
          if (v->vk != Value::Inst) break;
          const auto* i = static_cast<const Instruction*>(v);
          uint64_t j;
          if (i->op == Op::InsertElement && constantInt(i->ops[2], j) && j < v->type->n) {
            if (j == k) found = i->ops[1];
            v = i->ops[0];  // when not found: the lane is in the vector inserted into
          } else if (i->op == Op::ShuffleVector) {
            const auto* mask = static_cast<const Constant*>(i->ops[2]);
            const Value* lane = mask->ck == Constant::Aggregate ? mask->ops[k] : mask;
            uint64_t m;
            if (lane->vk == Value::Const && static_cast<const Constant*>(lane)->ck == Constant::Undef) {
              found = m_.constUndef(x->type);
              break;
            }
            if (!constantInt(lane, m)) break;
            unsigned na = i->ops[0]->type->n;
            v = m < na ? i->ops[0] : i->ops[1];
            k = m < na ? m : m - na;
          } else {
            break;
          }
          if (!found && (v != x->ops[0] || k != static_cast<const Constant*>(x->ops[1])->bits)) {
            x->ops[0] = v;  // the same lane, one step closer to its definition
            x->ops[1] = m_.constInt(x->ops[1]->type, int64_t(k));
            any = true;
          }
        }
        if (found) {
          replace(x, found);
          any = true;
          ++stats_.forwardedExtracts;
        }
      }
    if (!any) return false;
    applyReplacements();
    return true;
  }

  // F: removable instructions no live instruction reaches through its operands.
  bool sweepDead() {
    std::vector<char> live(n_, 0);
    std::vector<const Instruction*> work;
    auto mark = [&](const Instruction* i) {
      for (const Value* v : i->ops)
        if (unsigned k = id(v); k != ~0u && !live[k]) live[k] = 1, work.push_back(static_cast<const Instruction*>(v));
    };
    for (const BasicBlock* b : f_->blocks)
      for (const Instruction* i : b->insts)
        if (!removable(i)) mark(i);
    while (!work.empty()) {
      const Instruction* i = work.back();
      work.pop_back();
      mark(i);
    }
    bool changed = false;
    for (BasicBlock* b : f_->blocks) {
      auto dead = [&](Instruction* i) {
        if (!removable(i) || live[id(i)]) return false;
        ++(i->op == Op::Phi ? stats_.deadPhis : stats_.deadInsts);
        return true;
      };
      auto end = std::remove_if(b->insts.begin(), b->insts.end(), dead);
      changed |= end != b->insts.end();
      b->insts.erase(end, b->insts.end());
    }
    return changed;
  }

  Module& m_;
  Function* f_;
  CleanupStats stats_;
  unsigned long changes_ = 0;
  unsigned long clean_[6] = {~0ul, ~0ul, ~0ul, ~0ul, ~0ul, ~0ul};
  uint32_t ibase_ = 0;
  unsigned n_ = 0;
  std::vector<Value*> repl_;  // by instruction id: what the instruction is replaced by, or null
  bool pending_ = false;      // repl_ has entries
};

}  // namespace

CleanupStats cleanupFunction(Module& m, Function* f) {
  if (f->blocks.empty()) return {};
  return Cleaner(m, f).run();
}

std::string verifyFunction(const Function* f) {
  if (f->blocks.empty()) return "";
  std::unordered_set<const BasicBlock*> blocks(f->blocks.begin(), f->blocks.end());
  std::unordered_map<const Instruction*, const BasicBlock*> home;
  std::unordered_map<const BasicBlock*, std::vector<const BasicBlock*>> preds;
  for (const BasicBlock* b : f->blocks) {
    if (b->insts.empty() || !isTerminator(b->insts.back())) return "block without terminator";
    for (const Instruction* i : b->insts) home[i] = b;
    for (const BasicBlock* s : successors(b)) {
      if (!blocks.count(s)) return "branch to a block outside the function";
      preds[s].push_back(b);
    }
  }
  if (!preds[f->blocks[0]].empty()) return "entry block has predecessors";
  for (const BasicBlock* b : f->blocks) {
    bool phis = true;
    std::unordered_set<const Instruction*> before;
    for (size_t k = 0; k < b->insts.size(); ++k) {
      const Instruction* i = b->insts[k];
      if (isTerminator(i) != (k + 1 == b->insts.size())) return "terminator not exactly at the end of a block";
      if (i->op == Op::Phi) {
        if (!phis) return "phi after a non-phi";
        if (i->bbs.size() != i->ops.size()) return "phi with mismatched operand and block counts";
        std::vector<const BasicBlock*> want = preds[b], have(i->bbs.begin(), i->bbs.end());
        std::sort(want.begin(), want.end());
        std::sort(have.begin(), have.end());
        if (want != have) return "phi entries differ from the predecessor edges";
      } else {
        phis = false;
      }
      for (const Value* v : i->ops) {
        if (v->vk != Value::Inst) continue;
        auto h = home.find(static_cast<const Instruction*>(v));
        if (h == home.end()) return "operand defined outside the function";
        if (i->op != Op::Phi && h->second == b && !before.count(static_cast<const Instruction*>(v)))
          return "operand used before its definition in the same block";
      }
      before.insert(i);
    }
  }
  return "";
}

}  // namespace spirv2air
