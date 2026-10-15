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
struct Cfg {
  explicit Cfg(const Function* f) : blocks(f->blocks) {
    const size_t n = blocks.size();
    index.reserve(n);
    for (size_t b = 0; b < n; ++b) index[blocks[b]] = int(b);
    succ.resize(n);
    pred.resize(n);
    for (size_t b = 0; b < n; ++b)
      for (const BasicBlock* s : successors(blocks[b])) {
        int j = index.at(s);
        succ[b].push_back(j);
        pred[j].push_back(int(b));
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
    children.resize(n);
    for (int b : rpo)
      if (b != 0) children[idom[b]].push_back(b);
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
  std::vector<std::vector<int>> frontiers() const {
    std::vector<std::vector<int>> frontier(blocks.size());
    for (int b : rpo) {
      if (pred[b].size() < 2) continue;
      for (int p : pred[b])
        for (int r = p; po[r] >= 0 && r != idom[b]; r = idom[r]) frontier[r].push_back(b);
    }
    return frontier;
  }

  std::vector<BasicBlock*> blocks;
  std::unordered_map<const BasicBlock*, int> index;
  std::vector<std::vector<int>> succ, pred, children;
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
  using Preds = std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>;

  Preds predecessors() const {
    Preds preds;
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
    std::unordered_map<const Value*, const BasicBlock*> phiHome;
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
      for (size_t k = 0; k + 1 < c->insts.size(); ++k) phiHome[c->insts[k]] = c;
    }
    if (latches.empty()) return false;
    // The latch's phis may be read only by h's phis, through the entry for the latch.
    std::unordered_set<const BasicBlock*> bad;
    std::unordered_set<const BasicBlock*> latchSet(latches.begin(), latches.end());
    for (const BasicBlock* c : latches)
      for (const BasicBlock* b : preds[c]) {
        if (latchSet.count(b)) bad.insert(c);
      }
    for (const BasicBlock* b : f_->blocks)
      for (const Instruction* i : b->insts)
        for (size_t k = 0; k < i->ops.size(); ++k) {
          auto it = phiHome.find(i->ops[k]);
          if (it == phiHome.end()) continue;
          if (!(b == h && i->op == Op::Phi && i->bbs[k] == it->second)) bad.insert(it->second);
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
        auto home = phiHome.find(x);
        for (BasicBlock* b : ps) {
          Value* v = x;
          if (home != phiHome.end() && home->second == c) {
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
    std::unordered_map<const Value*, size_t> hidx;
    for (Instruction* i : h->insts) {
      if (i->op != Op::Phi) break;
      hidx[i] = hp.size();
      hp.push_back(i);
    }
    const size_t nv = hp.size();
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
          auto it = hidx.find(x);
          phi->ops.push_back(it == hidx.end() ? x : t.ev[it->second]);
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
    Cfg cfg(f_);
    const int hb = cfg.index.at(h);
    std::vector<char> hasPhi(cfg.blocks.size(), 0);
    {
      std::vector<std::vector<int>> frontier = cfg.frontiers();
      std::vector<int> work;
      for (const BasicBlock* d : targets) {
        int b = cfg.index.at(d);
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
          auto it = hidx.find(op);
          if (it != hidx.end() && cur[it->second]) op = cur[it->second];
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
            auto it = hidx.find(phi->ops[k]);
            if (it != hidx.end() && cur[it->second]) phi->ops[k] = cur[it->second];
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
          use = cfg.index.at(i->bbs[o]);
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
