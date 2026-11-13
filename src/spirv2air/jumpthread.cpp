// D8 jump threading of the Xenos dispatch loop. xenia wraps the guest control flow in a loop around a `switch (pc)`
// where every guest jump or label fall-through sets the next `pc` to a constant and continues. After promotion that is
// a header H of phis (`pc` and the carried state) ending in `switch pc`, entered from a latch that holds the phi of the
// next pc and of the carried state, and from the function's start with pc 0. The driver compiler pays for the
// loop-carried dispatch state, so where the next pc is known the edge goes straight to the case:
//   1  A latch C (only phis and `br H`) is bypassed: its predecessors branch to H, and H's phis take C's entry for
//      each of them.
//   2  An edge P -> H whose pc is a constant (or a select of two constants, with P ending in `br H`, which becomes a
//      `br` on the select's condition) is redirected to the block the switch picks, T. T's phis get an entry for P
//      (H's entry, with H's phis replaced by their entries for P).
//   3  SSA form is rebuilt for the values H's phis stood for: they are defined again at the start of every target T (a
//      phi whose entries are H's phi for the switch edges and H's entry for the threaded edges) and at the iterated
//      dominance frontier of H and the targets; uses take the reaching definition on the dominator tree (as in
//      PromoteMemToReg). Trivial and unused phis are left to cleanupFunction.
// Only edges change. Every instruction still runs on the same paths with the same values, and nothing is duplicated.
// Real backward guest jumps (non-constant pc) keep H, with fewer predecessors. All walks are iterative.
#include "spirv2air/jumpthread.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

const SmallVec<BasicBlock*, 2>& successors(const BasicBlock* b) {
  static const SmallVec<BasicBlock*, 2> none{};
  if (b->insts.empty()) return none;
  const Instruction* t = b->insts.back();
  return t->op == Op::Br || t->op == Op::Switch ? t->bbs : none;
}

bool constantInt(const Value* v, uint64_t& out) {
  if (v->vk != Value::Const) return false;
  const auto* c = static_cast<const Constant*>(v);
  if (c->ck != Constant::Int && c->ck != Constant::Null) return false;
  out = c->ck == Constant::Int ? c->bits : 0;
  return true;
}

size_t edgesTo(const BasicBlock* from, const BasicBlock* to) {
  size_t n = 0;
  for (const BasicBlock* s : successors(from)) n += s == to;
  return n;
}

// Dominators (Cooper, Harvey, Kennedy) and dominance frontiers of the blocks reachable from the entry. Blocks are
// numbered in layout order; edges are kept per terminator operand.
// Rows of block numbers in two flat arrays (compressed rows): row b is v[at[b] .. at[b + 1]). Filled in two passes, one
// to count and one to place, so building one allocates twice however many rows there are.
struct Rows {
  struct View {
    const int* first;
    const int* last;
    const int* begin() const { return first; }
    const int* end() const { return last; }
    size_t size() const { return size_t(last - first); }
    int operator[](size_t k) const { return first[k]; }
  };
  View operator[](size_t b) const { return {v.data() + at[b], v.data() + at[b + 1]}; }
  // Pass 1: count(b) once per element of row b; then close(); pass 2: put(b, x) in row order.
  void open(size_t n) { at.assign(n + 1, 0); }
  void count(size_t b) { ++at[b + 1]; }
  void close() {
    for (size_t b = 0; b + 1 < at.size(); ++b) at[b + 1] += at[b];
    v.resize(size_t(at.back()));
    fill.assign(at.begin(), at.end() - 1);
  }
  void put(size_t b, int x) { v[size_t(fill[b]++)] = x; }
  std::vector<int> at, v, fill;
};

struct Cfg {
  // For the verifier: the blocks are found through a map.
  explicit Cfg(const Function* f) : blocks(f->blocks) {
    index.reserve(blocks.size());
    for (size_t b = 0; b < blocks.size(); ++b) index[blocks[b]] = int(b);
    build();
  }
  // The blocks are numbered with tags (tag - base), which the caller must not overwrite while it uses the Cfg.
  Cfg(Module& m, const Function* f) : blocks(f->blocks), tagged(true) {
    base = m.allocTags(unsigned(blocks.size()));
    for (size_t b = 0; b < blocks.size(); ++b) blocks[b]->tag = base + uint32_t(b);
    build();
  }
  int indexOf(const BasicBlock* b) const { return tagged ? int(b->tag - base) : index.at(b); }

  void build() {
    const size_t n = blocks.size();
    succ.open(n);
    pred.open(n);
    for (size_t b = 0; b < n; ++b)
      for (const BasicBlock* s : successors(blocks[b])) succ.count(b), pred.count(size_t(indexOf(s)));
    succ.close();
    pred.close();
    for (size_t b = 0; b < n; ++b)
      for (const BasicBlock* s : successors(blocks[b])) {
        const int j = indexOf(s);
        succ.put(b, j);
        pred.put(size_t(j), int(b));
      }
    po.assign(n, -1);
    std::vector<char> seen(n, 0);
    std::vector<std::pair<int, size_t>> stack{{0, 0}};
    seen[0] = 1;
    while (!stack.empty()) {
      auto& [b, k] = stack.back();
      if (k < succ[b].size()) {
        int s = succ[b][k++];
        if (!seen[s]) seen[s] = 1, stack.push_back({s, 0});
      } else {
        po[b] = int(rpo.size());
        rpo.push_back(b);
        stack.pop_back();
      }
    }
    std::reverse(rpo.begin(), rpo.end());
    idom.assign(n, -1);
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
    children.open(n);
    for (int b : rpo)
      if (b != 0) children.count(size_t(idom[b]));
    children.close();
    for (int b : rpo)
      if (b != 0) children.put(size_t(idom[b]), b);
    // Dominator tree intervals for O(1) dominance tests.
    tin.assign(n, 0);
    tout.assign(n, 0);
    int clock = 0;
    std::vector<std::pair<int, size_t>> walk{{0, 0}};
    tin[0] = clock++;
    while (!walk.empty()) {
      auto& [b, k] = walk.back();
      if (k < children[b].size()) {
        int c = children[b][k++];
        tin[c] = clock++;
        walk.push_back({c, 0});
      } else {
        tout[b] = clock++;
        walk.pop_back();
      }
    }
  }

  bool reachable(int b) const { return po[b] >= 0; }
  bool dominates(int a, int b) const { return tin[a] <= tin[b] && tout[b] <= tout[a]; }

  // Block `b`'s frontier lists (only for the reachable blocks).
  Rows frontiers() const {
    Rows frontier;
    frontier.open(blocks.size());
    for (int pass = 0; pass < 2; ++pass) {
      for (int b : rpo) {
        if (pred[b].size() < 2) continue;
        for (int p : pred[b])
          for (int r = p; po[r] >= 0 && r != idom[b]; r = idom[r]) pass ? frontier.put(size_t(r), b) : frontier.count(size_t(r));
      }
      if (!pass) frontier.close();
    }
    return frontier;
  }

  std::vector<BasicBlock*> blocks;
  bool tagged = false;
  uint32_t base = 0;
  std::unordered_map<const BasicBlock*, int> index;  // only without tags
  Rows succ, pred, children;
  std::vector<int> po, rpo, idom, tin, tout;
};

class Threader {
 public:
  Threader(Module& m, Function* f, JumpThreadStats* stats) : m_(m), f_(f), stats_(stats) {}

  bool run() {
    std::vector<BasicBlock*> dispatchers;
    for (BasicBlock* b : f_->blocks)
      if (isDispatcher(b)) dispatchers.push_back(b);
    bool any = false;
    for (BasicBlock* h : dispatchers) {
      bool changed = dissolveLatches(h);
      changed |= thread(h);
      if (changed && stats_) ++stats_->dispatchers;
      any |= changed;
    }
    return any;
  }

 private:
  // The predecessors of each block, one per edge, in layout order. The blocks are numbered with tags (tag - base).
  struct Preds {
    uint32_t base = 0;
    std::vector<std::vector<BasicBlock*>> of;
    std::vector<BasicBlock*>& operator[](const BasicBlock* b) { return of[b->tag - base]; }
  };

  Preds predecessors() const {
    Preds preds;
    preds.base = m_.allocTags(unsigned(f_->blocks.size()));
    preds.of.resize(f_->blocks.size());
    for (size_t k = 0; k < f_->blocks.size(); ++k) f_->blocks[k]->tag = preds.base + uint32_t(k);
    for (BasicBlock* b : f_->blocks)
      for (const BasicBlock* s : successors(b)) preds[s].push_back(b);
    return preds;
  }

  static bool isDispatcher(const BasicBlock* h) {
    if (h->insts.size() < 2) return false;
    const Instruction* sw = h->insts.back();
    if (sw->op != Op::Switch || sw->ops.empty()) return false;
    bool selector = false;
    for (size_t k = 0; k + 1 < h->insts.size(); ++k) {
      if (h->insts[k]->op != Op::Phi) return false;
      selector |= h->insts[k] == sw->ops[0];
    }
    return selector;
  }

  static bool entryFor(const Instruction* phi, const BasicBlock* from, size_t& at) {
    auto it = std::find(phi->bbs.begin(), phi->bbs.end(), from);
    at = size_t(it - phi->bbs.begin());
    return it != phi->bbs.end();
  }

  static void eraseEntry(Instruction* phi, size_t k) {
    phi->ops.erase(phi->ops.begin() + long(k));
    phi->bbs.erase(phi->bbs.begin() + long(k));
  }

  // 1
  bool dissolveLatches(BasicBlock* h) {
    Preds preds = predecessors();
    // Candidates: a predecessor made of phis and `br h`, with the predecessors of its own.
    std::vector<BasicBlock*> latches;
    for (BasicBlock* c : preds[h]) {
      if (c == h || c->insts.empty() || preds[c].empty() || edgesTo(c, h) != 1) continue;
      const Instruction* t = c->insts.back();
      if (t->op != Op::Br || t->bbs.size() != 1) continue;
      bool ok = std::count(preds[h].begin(), preds[h].end(), c) == 1;
      for (size_t k = 0; k + 1 < c->insts.size(); ++k) ok &= c->insts[k]->op == Op::Phi;
      for (const BasicBlock* b : preds[c]) ok &= b != h && b != c && edgesTo(b, c) == 1 && edgesTo(b, h) == 0;
      if (!ok) continue;
      latches.push_back(c);
    }
    if (latches.empty()) return false;
    // The latches' phis are numbered with tags (tag - phiBase), and phiOwner[k] is the latch of phi k.
    unsigned nphis = 0;
    for (const BasicBlock* c : latches) nphis += unsigned(c->insts.size()) - 1;
    const uint32_t phiBase = m_.allocTags(nphis);
    std::vector<const Instruction*> phiList;
    std::vector<const BasicBlock*> phiOwner;
    phiList.reserve(nphis), phiOwner.reserve(nphis);
    for (const BasicBlock* c : latches)
      for (size_t k = 0; k + 1 < c->insts.size(); ++k) {
        c->insts[k]->tag = phiBase + unsigned(phiList.size());
        phiList.push_back(c->insts[k]);
        phiOwner.push_back(c);
      }
    // The latch a value is a phi of, or null.
    auto phiHome = [&](const Value* v) -> const BasicBlock* {
      const unsigned k = v->tag - phiBase;
      return v->vk == Value::Inst && k < nphis && phiList[k] == v ? phiOwner[k] : nullptr;
    };
    // The latch's phis may be read only by h's phis, through the entry for the latch.
    std::unordered_set<const BasicBlock*> bad;
    std::unordered_set<const BasicBlock*> latchSet(latches.begin(), latches.end());
    for (const BasicBlock* c : latches)
      for (const BasicBlock* b : preds[c]) {
        if (latchSet.count(b)) bad.insert(c);
      }
    if (nphis)
      for (const BasicBlock* b : f_->blocks)
        for (const Instruction* i : b->insts)
          for (size_t k = 0; k < i->ops.size(); ++k) {
            const BasicBlock* home = phiHome(i->ops[k]);
            if (!home) continue;
            if (!(b == h && i->op == Op::Phi && i->bbs[k] == home)) bad.insert(home);
          }
    std::unordered_set<const BasicBlock*> removed;
    for (BasicBlock* c : latches) {
      if (bad.count(c)) continue;
      const std::vector<BasicBlock*>& ps = preds[c];
      for (Instruction* phi : h->insts) {
        if (phi->op != Op::Phi) break;
        size_t at;
        if (!entryFor(phi, c, at)) continue;
        Value* x = phi->ops[at];
        eraseEntry(phi, at);
        const BasicBlock* home = phiHome(x);
        for (BasicBlock* b : ps) {
          Value* v = x;
          if (home == c) {
            size_t k;
            entryFor(static_cast<const Instruction*>(x), b, k);
            v = static_cast<const Instruction*>(x)->ops[k];
          }
          phi->ops.push_back(v);
          phi->bbs.push_back(b);
        }
      }
      for (BasicBlock* b : ps)
        for (BasicBlock*& s : b->insts.back()->bbs)
          if (s == c) s = h;
      removed.insert(c);
      if (stats_) ++stats_->dissolvedLatches;
    }
    if (removed.empty()) return false;
    f_->blocks.erase(
        std::remove_if(f_->blocks.begin(), f_->blocks.end(), [&](BasicBlock* b) { return removed.count(b) != 0; }),
        f_->blocks.end());
    return true;
  }

  struct Thread {
    BasicBlock* p;
    BasicBlock* d1;
    BasicBlock* d2 = nullptr;  // with cond: the false side
    Value* cond = nullptr;
    std::vector<Value*> ev;  // h's phi entries for p
  };

  // 2, 3
  bool thread(BasicBlock* h) {
    Preds preds = predecessors();
    Instruction* sw = h->insts.back();
    std::vector<Instruction*> hp;
    for (Instruction* i : h->insts) {
      if (i->op != Op::Phi) break;
      hp.push_back(i);
    }
    const size_t nv = hp.size();
    // h's phis are numbered with tags (tag - hBase); hIndex is a value's number, or npos.
    const uint32_t hBase = m_.allocTags(unsigned(nv));
    for (size_t v = 0; v < nv; ++v) hp[v]->tag = hBase + uint32_t(v);
    constexpr size_t npos = ~size_t(0);
    auto hIndex = [&](const Value* v) -> size_t {
      const size_t k = v->tag - hBase;
      return v->vk == Value::Inst && k < nv && hp[k] == v ? k : npos;
    };
    Instruction* pcPhi = static_cast<Instruction*>(sw->ops[0]);
    auto destOf = [&](uint64_t c) {
      for (size_t k = 1; k < sw->bbs.size(); ++k) {
        uint64_t cv = 0;
        if (k < sw->ops.size() && constantInt(sw->ops[k], cv) && cv == c) return sw->bbs[k];
      }
      return sw->bbs[0];
    };
    // Destination phis must have an entry for h to take values from.
    auto usable = [&](const BasicBlock* p, BasicBlock* d) {
      if (d == h || d == p || edgesTo(p, d) != 0) return false;
      for (const Instruction* phi : d->insts) {
        if (phi->op != Op::Phi) break;
        size_t at;
        if (!entryFor(phi, h, at)) return false;
      }
      return true;
    };
    std::vector<Thread> threads;
    std::unordered_set<const BasicBlock*> seen;
    for (BasicBlock* p : preds[h]) {
      if (p == h || edgesTo(p, h) != 1 || !seen.insert(p).second) continue;
      size_t k;
      if (!entryFor(pcPhi, p, k)) continue;
      Value* v = pcPhi->ops[k];
      Thread t;
      t.p = p;
      uint64_t c, c1, c2;
      Instruction* sel = v->vk == Value::Inst ? static_cast<Instruction*>(v) : nullptr;
      if (constantInt(v, c)) {
        t.d1 = destOf(c);
        if (!usable(p, t.d1)) continue;
      } else if (sel && sel->op == Op::Select && sel->ops.size() == 3 && constantInt(sel->ops[1], c1) &&
                 constantInt(sel->ops[2], c2) && sel->ops[0]->type->kind == Type::Int && sel->ops[0]->type->n == 1 &&
                 p->insts.back()->op == Op::Br && p->insts.back()->bbs.size() == 1) {
        t.d1 = destOf(c1);
        t.d2 = destOf(c2);
        t.cond = sel->ops[0];
        if (t.d1 == t.d2) t.d2 = nullptr, t.cond = nullptr;
        if (!usable(p, t.d1) || (t.d2 && !usable(p, t.d2))) continue;
      } else {
        continue;
      }
      threads.push_back(std::move(t));
    }
    if (threads.empty()) return false;

    // Redirect the edges.
    std::unordered_map<const BasicBlock*, std::unordered_map<const BasicBlock*, size_t>> edgeThread;
    std::vector<BasicBlock*> targets;
    std::unordered_set<const BasicBlock*> isTarget;
    for (size_t ti = 0; ti < threads.size(); ++ti) {
      Thread& t = threads[ti];
      for (Instruction* phi : hp) {
        size_t at;
        entryFor(phi, t.p, at);
        t.ev.push_back(phi->ops[at]);
      }
      for (BasicBlock* d : {t.d1, t.d2}) {
        if (!d) continue;
        for (Instruction* phi : d->insts) {
          if (phi->op != Op::Phi) break;
          size_t at;
          entryFor(phi, h, at);
          Value* x = phi->ops[at];
          const size_t xi = hIndex(x);
          phi->ops.push_back(xi == npos ? x : t.ev[xi]);
          phi->bbs.push_back(t.p);
        }
        edgeThread[t.p][d] = ti;
        if (isTarget.insert(d).second) targets.push_back(d);
      }
      Instruction* term = t.p->insts.back();
      if (t.cond) {
        term->ops = {t.cond};
        term->bbs = {t.d1, t.d2};
        if (stats_) ++stats_->splitSelects;
      } else {
        for (BasicBlock*& s : term->bbs)
          if (s == h) s = t.d1;
      }
      for (Instruction* phi : hp) {
        size_t at;
        entryFor(phi, t.p, at);
        eraseEntry(phi, at);
      }
      if (stats_) ++stats_->threadedEdges;
    }

    // Rebuild SSA form on the new CFG.
    Cfg cfg(m_, f_);
    const int hb = cfg.indexOf(h);
    std::vector<char> hasPhi(cfg.blocks.size(), 0);
    {
      const Rows frontier = cfg.frontiers();
      std::vector<int> work;
      for (const BasicBlock* d : targets) {
        int b = cfg.indexOf(d);
        hasPhi[b] = 1;
        work.push_back(b);
      }
      if (cfg.reachable(hb)) work.push_back(hb);
      std::vector<char> queued(cfg.blocks.size(), 0);
      for (int b : work) queued[b] = 1;
      while (!work.empty()) {
        int b = work.back();
        work.pop_back();
        for (int d : frontier[b]) {
          if (d != hb) hasPhi[d] = 1;
          if (!queued[d]) queued[d] = 1, work.push_back(d);
        }
      }
    }
    // The new phis: one per variable at every block marked. An entry is the variable itself (resolved to its reaching
    // definition at the end of the incoming block below) except on a threaded edge, where it is h's entry for it.
    std::vector<std::vector<Instruction*>> newPhi(cfg.blocks.size());
    for (size_t b = 0; b < cfg.blocks.size(); ++b) {
      if (!hasPhi[b]) continue;
      BasicBlock* blk = cfg.blocks[b];
      std::vector<Instruction*> mine;
      for (size_t v = 0; v < nv; ++v) {
        Instruction* phi = m_.newInst(Op::Phi, hp[v]->type);
        phi->flags = hp[v]->flags;
        for (int q : cfg.pred[b]) {
          const BasicBlock* from = cfg.blocks[q];
          Value* x = hp[v];
          auto e = edgeThread.find(from);
          if (e != edgeThread.end()) {
            auto d = e->second.find(blk);
            if (d != e->second.end()) x = threads[d->second].ev[v];
          }
          phi->ops.push_back(x);
          phi->bbs.push_back(cfg.blocks[q]);
        }
        mine.push_back(phi);
      }
      blk->insts.insert(blk->insts.begin(), mine.begin(), mine.end());
      newPhi[b] = std::move(mine);
      if (stats_) stats_->phis += unsigned(nv);
    }
    // Renaming walk over the dominator tree.
    std::vector<Value*> cur(nv, nullptr);
    struct Frame {
      int b;
      size_t child;
      std::vector<Value*> saved;
      bool defines;
    };
    std::vector<Frame> stack;
    auto enter = [&](int b) {
      Frame fr{b, 0, {}, hasPhi[b] || b == hb};
      if (fr.defines) {
        fr.saved = cur;
        for (size_t v = 0; v < nv; ++v) cur[v] = b == hb ? static_cast<Value*>(hp[v]) : newPhi[b][v];
      }
      BasicBlock* blk = cfg.blocks[b];
      for (Instruction* i : blk->insts) {
        if (i->op == Op::Phi) continue;
        for (Value*& op : i->ops) {
          const size_t oi = hIndex(op);
          if (oi != npos && cur[oi]) op = cur[oi];
        }
      }
      // The entries of the successors' phis for this block read the values at its end.
      std::unordered_set<int> done;
      for (int s : cfg.succ[b]) {
        if (!done.insert(s).second) continue;
        for (Instruction* phi : cfg.blocks[s]->insts) {
          if (phi->op != Op::Phi) break;
          for (size_t k = 0; k < phi->ops.size(); ++k) {
            if (phi->bbs[k] != blk) continue;
            const size_t oi = hIndex(phi->ops[k]);
            if (oi != npos && cur[oi]) phi->ops[k] = cur[oi];
          }
        }
      }
      stack.push_back(std::move(fr));
    };
    enter(0);
    while (!stack.empty()) {
      Frame& fr = stack.back();
      if (fr.child < cfg.children[fr.b].size()) {
        int c = cfg.children[fr.b][fr.child++];
        enter(c);
      } else {
        if (fr.defines) cur = std::move(fr.saved);
        stack.pop_back();
      }
    }
    return true;
  }

  Module& m_;
  Function* f_;
  JumpThreadStats* stats_;
};

}  // namespace

bool jumpThread(Module& m, Function* f, JumpThreadStats* stats) {
  if (f->blocks.empty()) return false;
  return Threader(m, f, stats).run();
}

std::string verifyDominance(const Function* f) {
  if (f->blocks.empty()) return "";
  Cfg cfg(f);
  std::unordered_map<const Instruction*, std::pair<int, size_t>> home;
  for (size_t b = 0; b < cfg.blocks.size(); ++b)
    for (size_t k = 0; k < cfg.blocks[b]->insts.size(); ++k) home[cfg.blocks[b]->insts[k]] = {int(b), k};
  for (size_t b = 0; b < cfg.blocks.size(); ++b) {
    if (!cfg.reachable(int(b))) continue;
    const BasicBlock* blk = cfg.blocks[b];
    for (size_t k = 0; k < blk->insts.size(); ++k) {
      const Instruction* i = blk->insts[k];
      for (size_t o = 0; o < i->ops.size(); ++o) {
        const Value* v = i->ops[o];
        if (v->vk != Value::Inst) continue;
        auto h = home.find(static_cast<const Instruction*>(v));
        if (h == home.end()) return "operand defined outside the function";
        int use = int(b);
        size_t at = k;
        if (i->op == Op::Phi) {
          use = cfg.indexOf(i->bbs[o]);
          if (!cfg.reachable(use)) continue;
          at = ~size_t(0);
        }
        if (!cfg.reachable(h->second.first)) return "operand defined in an unreachable block";
        bool ok = h->second.first == use ? h->second.second < at : cfg.dominates(h->second.first, use);
        if (!ok) return "operand does not dominate its use";
      }
    }
  }
  return "";
}

}  // namespace spirv2air
