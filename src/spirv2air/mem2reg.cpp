// SSA promotion, as in Cytron et al. and LLVM's PromoteMemToReg: dominators by Cooper, Harvey and Kennedy, phis at the
// iterated dominance frontier of each slot's stores, then one renaming walk over the dominator tree. Phis that turn out
// to be trivial (one incoming value besides themselves) or unused are dropped afterwards. Every walk is iterative: the
// largest corpus shader has over 6,000 blocks. Before that, arrays and structs addressed only with constant indices are
// split into one alloca per element (scalar replacement of aggregates), so their elements can be promoted too.
#include "spirv2air/mem2reg.h"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

bool constantIndex(const Value* v, uint64_t& out) {
  if (v->vk != Value::Const) return false;
  const auto* c = static_cast<const Constant*>(v);
  if (c->ck != Constant::Int && c->ck != Constant::Null) return false;
  out = c->ck == Constant::Int ? c->bits : 0;
  return true;
}

// Splits each entry-block array or struct alloca whose only uses are GEPs `0, c, ...` with constant indices into one
// alloca per element; a GEP `0, c` becomes element c's alloca and a longer one a GEP into it. Repeats until nothing
// splits, so nested aggregates split level by level.
void splitAggregates(Module& m, Function* f) {
  BasicBlock* entry = f->blocks[0];
  for (bool changed = true; changed;) {
    changed = false;
    // The aggregate allocas get ids 0..A-1 (tag - base), the GEPs on them the ids after them.
    std::vector<Instruction*> aggregates;
    for (Instruction* i : entry->insts)
      if (i->op == Op::Alloca && (i->srcElemTy->kind == Type::Array || i->srcElemTy->kind == Type::Struct))
        aggregates.push_back(i);
    if (aggregates.empty()) break;
    unsigned total = 0;
    for (const BasicBlock* b : f->blocks) total += unsigned(b->insts.size());
    const uint32_t base = m.allocTags(total);
    const unsigned A = unsigned(aggregates.size());
    unsigned next = A;
    for (unsigned a = 0; a < A; ++a) aggregates[a]->tag = base + a;
    auto aggregateOf = [&](const Value* v) -> int {
      unsigned k = v->tag - base;
      return k < A && v->vk == Value::Inst && aggregates[k] == v ? int(k) : -1;
    };
    std::vector<std::vector<Instruction*>> geps(A);
    std::vector<char> unsplittable(A, 0);
    for (BasicBlock* b : f->blocks)
      for (Instruction* i : b->insts)
        for (size_t k = 0; k < i->ops.size(); ++k) {
          int a = aggregateOf(i->ops[k]);
          if (a < 0) continue;
          const Type* t = aggregates[a]->srcElemTy;
          uint64_t first = 1, c = 0;
          bool ok = i->op == Op::GEP && k == 0 && i->ops.size() >= 3 && constantIndex(i->ops[1], first) && first == 0 &&
                    constantIndex(i->ops[2], c) && c < (t->kind == Type::Array ? t->n : t->elems.size());
          if (ok)
            geps[a].push_back(i), i->tag = base + next++;
          else
            unsplittable[a] = 1;
        }
    std::vector<Value*> repl(next, nullptr);
    std::vector<char> dead(next, 0);
    std::vector<Instruction*> insts;
    bool nested = false;  // an element is itself an aggregate, which the next round may split
    for (Instruction* i : entry->insts) {
      insts.push_back(i);
      int a = aggregateOf(i);
      if (a < 0 || geps[a].empty() || unsplittable[a]) continue;
      std::unordered_map<uint64_t, Instruction*> elements;
      for (Instruction* g : geps[a]) {
        uint64_t c = 0;
        constantIndex(g->ops[2], c);
        Type* t = i->srcElemTy->kind == Type::Array ? i->srcElemTy->elem : i->srcElemTy->elems[c];
        Instruction*& e = elements[c];
        if (!e) {
          e = m.newInst(Op::Alloca, m.ptrTy(t, 0));
          e->srcElemTy = t;
          e->align = i->align;
          e->ops = {m.constInt(m.intTy(32), 1)};
          insts.push_back(e);
          nested |= t->kind == Type::Array || t->kind == Type::Struct;
        }
        if (g->ops.size() == 3) {
          repl[g->tag - base] = e;
          dead[g->tag - base] = 1;
        } else {
          g->srcElemTy = t;
          g->ops.erase(g->ops.begin() + 2);
          g->ops[0] = e;
          g->ops[1] = m.constNull(g->ops[1]->type);
        }
      }
      dead[i->tag - base] = 1;
      changed = true;
    }
    entry->insts = std::move(insts);
    if (!changed) break;
    // Instructions created above (the element allocas) have tag 0, which is outside [base, base + next).
    auto idOf = [&](const Value* v) { return v->tag - base < next && v->vk == Value::Inst ? v->tag - base : ~0u; };
    for (BasicBlock* b : f->blocks) {
      std::vector<Instruction*>& list = b->insts;
      list.erase(std::remove_if(list.begin(), list.end(),
                                [&](Instruction* i) {
                                  unsigned k = idOf(i);
                                  return k != ~0u && dead[k];
                                }),
                 list.end());
      for (Instruction* i : list)
        for (Value*& v : i->ops)
          if (unsigned k = idOf(v); k != ~0u && repl[k]) v = repl[k];
    }
    changed = nested;
  }
}

}  // namespace

void promoteAllocas(Module& m, Function* f) {
  if (f->blocks.empty()) return;
  splitAggregates(m, f);
  const std::vector<BasicBlock*>& blocks = f->blocks;
  const size_t n = blocks.size();

  // Slots: entry-block allocas used only as the address of whole-value, non-volatile loads and stores.
  // Dense ids (tag - base) for what the later tables are indexed by: the entry block's allocas first, in order, then
  // the loads and stores of slots as they are met. Phis made below get ids after all instructions.
  std::vector<Instruction*> allocas;
  for (Instruction* i : blocks[0]->insts)
    if (i->op == Op::Alloca) allocas.push_back(i);
  if (allocas.empty()) return;
  const unsigned A = unsigned(allocas.size());
  unsigned ninsts = 0;
  for (const BasicBlock* b : blocks) ninsts += unsigned(b->insts.size());
  uint32_t base = m.allocTags(ninsts);
  unsigned nextId = A;
  for (unsigned a = 0; a < A; ++a) allocas[a]->tag = base + a;
  // The index of the entry-block alloca a value is, or -1.
  auto allocaOf = [&](const Value* v) -> int {
    unsigned k = v->tag - base;
    return k < A && v->vk == Value::Inst && allocas[k] == v ? int(k) : -1;
  };
  std::vector<char> promotable(A, 1);
  for (BasicBlock* b : blocks)
    for (Instruction* i : b->insts)
      for (size_t k = 0; k < i->ops.size(); ++k) {
        int a = allocaOf(i->ops[k]);
        if (a < 0) continue;
        Type* t = allocas[a]->srcElemTy;
        bool whole =
            (i->op == Op::Load && k == 0 && i->type == t) || (i->op == Op::Store && k == 1 && i->ops[0]->type == t);
        if (!whole || i->isVolatile) promotable[a] = 0;
      }
  std::vector<Instruction*> slots;
  std::vector<int> slotIndex(A, -1);  // by alloca index
  for (unsigned a = 0; a < A; ++a)
    if (promotable[a]) slotIndex[a] = int(slots.size()), slots.push_back(allocas[a]);
  if (slots.empty()) return;
  const size_t nslots = slots.size();
  // The slot a value is the address of, or -1.
  auto slot = [&](const Value* v) -> int {
    int a = allocaOf(v);
    return a < 0 ? -1 : slotIndex[a];
  };

  // CFG. Edges are kept per terminator operand, so a switch with two cases to one block gives two edges, and that
  // block's phis two entries for it, as LLVM requires.
  const uint32_t bbase = m.allocTags(unsigned(n));
  for (size_t b = 0; b < n; ++b) blocks[b]->tag = bbase + uint32_t(b);
  std::vector<std::vector<int>> succ(n), pred(n);
  for (size_t b = 0; b < n; ++b) {
    if (blocks[b]->insts.empty()) continue;
    const Instruction* t = blocks[b]->insts.back();
    if (t->op != Op::Br && t->op != Op::Switch) continue;
    for (BasicBlock* s : t->bbs) {
      int j = int(s->tag - bbase);
      succ[b].push_back(j);
      pred[j].push_back(int(b));
    }
  }

  // Postorder from the entry; blocks it never reaches keep po == -1.
  std::vector<int> post, po(n, -1);
  {
    std::vector<char> seen(n, 0);
    std::vector<std::pair<int, size_t>> stack{{0, 0}};
    seen[0] = 1;
    while (!stack.empty()) {
      auto& [b, k] = stack.back();
      if (k < succ[b].size()) {
        int s = succ[b][k++];
        if (!seen[s]) seen[s] = 1, stack.push_back({s, 0});
      } else {
        po[b] = int(post.size());
        post.push_back(b);
        stack.pop_back();
      }
    }
  }
  const std::vector<int> rpo(post.rbegin(), post.rend());

  std::vector<int> idom(n, -1);
  idom[0] = 0;
  for (bool changed = true; changed;) {
    changed = false;
    for (int b : rpo) {
      if (b == 0) continue;
      int d = -1;
      for (int p : pred[b]) {
        if (idom[p] < 0) continue;
        if (d < 0) {
          d = p;
          continue;
        }
        int x = p;
        while (x != d) {
          while (po[x] < po[d]) x = idom[x];
          while (po[d] < po[x]) d = idom[d];
        }
      }
      if (idom[b] != d) idom[b] = d, changed = true;
    }
  }

  std::vector<std::vector<int>> frontier(n), children(n);
  for (int b : rpo) {
    if (b != 0) children[idom[b]].push_back(b);
    if (pred[b].size() < 2) continue;
    for (int p : pred[b])
      for (int r = p; po[r] >= 0 && r != idom[b]; r = idom[r]) frontier[r].push_back(b);
  }

  // Phis at the iterated dominance frontier of each slot's reachable stores.
  std::vector<std::vector<std::pair<unsigned, Instruction*>>> phis(n);
  {
    std::vector<std::vector<int>> defs(nslots);
    for (int b : rpo)
      for (const Instruction* i : blocks[b]->insts)
        if (i->op == Op::Store) {
          int a = slot(i->ops[1]);
          if (a >= 0 && (defs[a].empty() || defs[a].back() != b)) defs[a].push_back(b);
        }
    std::vector<size_t> placed(n, ~size_t(0)), queued(n, ~size_t(0));
    for (size_t a = 0; a < nslots; ++a) {
      std::vector<int> work = defs[a];
      for (int b : work) queued[b] = a;
      while (!work.empty()) {
        int b = work.back();
        work.pop_back();
        for (int d : frontier[b]) {
          if (placed[d] == a) continue;
          placed[d] = a;
          phis[d].push_back({unsigned(a), m.newInst(Op::Phi, slots[a]->srcElemTy)});
          if (queued[d] != a) queued[d] = a, work.push_back(d);
        }
      }
    }
  }

  // The phis get the ids after the existing instructions (their own tag range).
  unsigned nphis = 0;
  for (auto& list : phis) nphis += unsigned(list.size());
  const uint32_t pbase = m.allocTags(nphis);
  const unsigned nIds = ninsts + nphis;
  {
    unsigned p = 0;
    for (auto& list : phis)
      for (auto& [a, phi] : list) phi->tag = pbase + p++;
  }
  // Renaming walk over the dominator tree.
  std::vector<Value*> repl(nIds, nullptr);
  std::vector<char> dead(nIds, 0);
  auto idOf = [&](const Value* v) -> unsigned {
    if (v->vk != Value::Inst) return ~0u;
    if (unsigned k = v->tag - base; k < nextId) return k;
    if (unsigned k = v->tag - pbase; k < nphis) return ninsts + k;
    return ~0u;
  };
  auto number = [&](Instruction* i) -> unsigned {
    if (i->tag - base >= nextId) i->tag = base + nextId++;
    return i->tag - base;
  };
  auto resolve = [&](Value* v) {
    for (unsigned k = idOf(v); k != ~0u && repl[k]; k = idOf(v)) v = repl[k];
    return v;
  };
  std::vector<std::vector<Value*>> values(nslots);
  auto current = [&](unsigned a) -> Value* {
    return values[a].empty() ? m.constUndef(slots[a]->srcElemTy) : values[a].back();
  };
  auto slotOfUse = [&](const Instruction* i) -> int {
    if (i->op != Op::Load && i->op != Op::Store) return -1;
    return slot(i->ops[i->op == Op::Load ? 0 : 1]);
  };
  std::vector<std::vector<unsigned>> pushed(n);
  std::vector<std::pair<int, bool>> walk{{0, false}};  // block, leaving
  while (!walk.empty()) {
    auto [b, leaving] = walk.back();
    walk.pop_back();
    if (leaving) {
      for (unsigned a : pushed[b]) values[a].pop_back();
      continue;
    }
    walk.push_back({b, true});
    for (auto& [a, phi] : phis[b]) values[a].push_back(phi), pushed[b].push_back(a);
    for (Instruction* i : blocks[b]->insts) {
      int a = slotOfUse(i);
      if (a < 0) continue;
      if (i->op == Op::Load) {
        repl[number(i)] = current(unsigned(a));
      } else {
        values[a].push_back(resolve(i->ops[0]));
        pushed[b].push_back(unsigned(a));
      }
      dead[number(i)] = 1;
    }
    for (int s : succ[b])
      for (auto& [a, phi] : phis[s]) phi->ops.push_back(current(a)), phi->bbs.push_back(blocks[b]);
    for (auto c = children[b].rbegin(); c != children[b].rend(); ++c) walk.push_back({*c, false});
  }
  // Unreachable blocks: their loads read undef, and they still need entries in the phis of reachable successors.
  for (size_t b = 0; b < n; ++b) {
    if (po[b] >= 0) continue;
    for (Instruction* i : blocks[b]->insts) {
      int a = slotOfUse(i);
      if (a < 0) continue;
      if (i->op == Op::Load) repl[number(i)] = m.constUndef(slots[a]->srcElemTy);
      dead[number(i)] = 1;
    }
    for (int s : succ[b])
      for (auto& [a, phi] : phis[s])
        phi->ops.push_back(m.constUndef(slots[a]->srcElemTy)), phi->bbs.push_back(blocks[b]);
  }
  for (Instruction* a : slots) dead[idOf(a)] = 1;

  // Drop trivial phis until none is left; then one walk resolves every operand, removes the dead instructions and
  // notes which inserted phis something else uses.
  for (bool changed = true; changed;) {
    changed = false;
    for (auto& list : phis)
      for (auto& [a, phi] : list) {
        if (dead[idOf(phi)]) continue;
        Value* same = nullptr;
        bool trivial = true;
        for (Value*& v : phi->ops) {
          v = resolve(v);
          if (v == phi || v == same) continue;
          if (same) trivial = false;
          same = v;
        }
        if (!trivial) continue;
        repl[idOf(phi)] = same ? same : m.constUndef(phi->type);
        dead[idOf(phi)] = 1;
        changed = true;
      }
  }
  std::vector<char> live(nIds, 0);
  std::vector<Instruction*> work;
  auto use = [&](Value* v) {
    unsigned k = idOf(v);
    if (k != ~0u && k >= ninsts && !dead[k] && !live[k]) live[k] = 1, work.push_back(static_cast<Instruction*>(v));
  };
  for (BasicBlock* b : blocks) {
    std::vector<Instruction*>& insts = b->insts;
    size_t kept = 0;
    for (Instruction* i : insts) {
      if (unsigned k = idOf(i); k != ~0u && dead[k]) continue;
      for (Value*& v : i->ops) use(v = resolve(v));
      insts[kept++] = i;
    }
    insts.resize(kept);
  }
  // Keep only the inserted phis something else uses.
  while (!work.empty()) {
    Instruction* phi = work.back();
    work.pop_back();
    for (Value*& v : phi->ops) use(v = resolve(v));
  }

  for (size_t b = 0; b < n; ++b) {
    if (phis[b].empty()) continue;
    std::vector<Instruction*>& insts = blocks[b]->insts;
    std::vector<Instruction*> front;
    for (auto& [a, phi] : phis[b])
      if (live[idOf(phi)]) front.push_back(phi);
    insts.insert(insts.begin(), front.begin(), front.end());
  }
}

}  // namespace spirv2air
