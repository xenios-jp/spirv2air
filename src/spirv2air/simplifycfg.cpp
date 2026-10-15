// D7 control flow. Two transformations, repeated until none applies:
//   A  an empty block (only `br M`, no phis) is bypassed: its predecessors branch to M, and M's phis take the
//      value of the entry for the empty block once per predecessor. Skipped when a predecessor already has an edge
//      to M (the phi would need two entries for one predecessor, with possibly different values; the if-conversion
//      below handles that shape with a select).
//   B  if-conversion of triangles (H: br c, T, M; T: br M) and diamonds (H: br c, X, Y; X, Y: br M) whose arms are
//      single-predecessor blocks of speculatable instructions: the arms' instructions move to the end of H, and a
//      phi of M with entries for the arm(s) and H becomes a select on c in H. H then branches straight to M.
// Speculatable: an instruction without side effects that cannot trap or read memory (no loads, no integer division,
// only calls that touch no memory). It then runs on paths where it did not before, but its value is used only through
// the select that picks it when the original path ran, so nothing observable changes. Folding the main loop's `pc`
// switch is cleanupFunction's job and has finished by the time this runs.
#include "spirv2air/simplifycfg.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

// Instructions moved into the branching block, both arms together. More would make the guarded path's work run
// unconditionally for little gain in blocks.
constexpr size_t kMaxSpeculated = 8;
constexpr uint8_t kIntEq = 32;  // icmp eq

const decltype(Instruction::bbs)& successors(const BasicBlock* b) {
  static const decltype(Instruction::bbs) none{};
  if (b->insts.empty()) return none;
  const Instruction* t = b->insts.back();
  return t->op == Op::Br || t->op == Op::Switch ? t->bbs : none;
}

bool hasFunctionAttr(const AttrList* l, const Attr& a) {
  if (!l) return false;
  for (const auto& [index, set] : l->sets)
    if (index == AttrList::FunctionIndex && std::find(set->begin(), set->end(), a) != set->end()) return true;
  return false;
}

// A call that reads and writes no memory and has no effect: readnone, or readonly + argmemonly with no pointer among
// its arguments (the bit-field intrinsics: integers in, an integer out). Convergent calls (texture, derivatives)
// depend on which threads run them and stay where they are.
bool pureCall(const Instruction* i) {
  static const Attr readnone = Attr::get("readnone"), readonly = Attr::get("readonly"),
                    argmemonly = Attr::get("argmemonly"), convergent = Attr::get("convergent");
  if (i->ops.empty() || i->ops[0]->vk != Value::Func) return false;
  const AttrList* fa = static_cast<const Function*>(i->ops[0])->attrs;
  auto has = [&](const Attr& a) { return hasFunctionAttr(fa, a) || hasFunctionAttr(i->attrs, a); };
  if (has(convergent)) return false;
  if (has(readnone)) return true;
  if (!has(readonly) || !has(argmemonly)) return false;
  for (size_t k = 1; k < i->ops.size(); ++k)
    if (i->ops[k]->type->kind == Type::Pointer) return false;
  return true;
}

bool speculatable(const Instruction* i) {
  switch (i->op) {
    case Op::BinOp:
      // Integer division and remainder trap on zero (and on overflow); float ops share their opcodes.
      return !(i->type->isIntOrIntVector() && i->sub >= UDiv && i->sub <= SRem);
    case Op::Cast:
    case Op::GEP:
    case Op::ExtractElement:
    case Op::InsertElement:
    case Op::ShuffleVector:
    case Op::ExtractValue:
    case Op::InsertValue:
    case Op::Cmp:
    case Op::Select:
    case Op::FNeg:
      return true;
    case Op::Call:
      return pureCall(i);
    default:
      return false;
  }
}

// Types a select can stand in for a phi on.
bool selectable(const Type* t) {
  switch (t->kind) {
    case Type::Half:
    case Type::Float:
    case Type::Double:
    case Type::Int:
    case Type::Pointer:
    case Type::Vector:
      return true;
    default:
      return false;
  }
}

class Simplifier {
 public:
  Simplifier(Module& m, Function* f, SimplifyCfgStats* stats) : m_(m), f_(f), stats_(stats) {}

  bool run() {
    bool any = false;
    for (bool changed = true; changed;) {
      changed = lowerSwitches();
      changed |= forwardEmpty();
      changed |= ifConvert();
      any |= changed;
    }
    return any;
  }

 private:
  using Preds = std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>;

  // One entry per edge, in block order.
  Preds predecessors() const {
    Preds preds;
    for (BasicBlock* b : f_->blocks)
      for (const BasicBlock* s : successors(b)) preds[s].push_back(b);
    return preds;
  }

  static size_t edgesTo(const BasicBlock* from, const BasicBlock* to) {
    size_t n = 0;
    for (const BasicBlock* s : successors(from)) n += s == to;
    return n;
  }

  // The index of the phi's entry for `from`, or false.
  static bool entryFor(const Instruction* phi, const BasicBlock* from, size_t& at) {
    auto it = std::find(phi->bbs.begin(), phi->bbs.end(), from);
    at = size_t(it - phi->bbs.begin());
    return it != phi->bbs.end();
  }

  static void eraseEntry(Instruction* phi, size_t k) {
    phi->ops.erase(phi->ops.begin() + long(k));
    phi->bbs.erase(phi->bbs.begin() + long(k));
  }

  // C
  bool lowerSwitches() {
    Preds preds = predecessors();
    std::unordered_map<const BasicBlock*, BasicBlock*> placeAfter;
    std::unordered_set<const BasicBlock*> done;
    bool any = false;
    const std::vector<BasicBlock*> blocks = f_->blocks;  // addBlock below grows f_->blocks
    for (BasicBlock* h : blocks) {
      Instruction* sw = h->insts.back();
      if (sw->op != Op::Switch) continue;
      BasicBlock* join = sw->bbs[0];
      if (join == h || done.count(join)) continue;
      // The arms: distinct single-predecessor blocks that only compute and branch to the join. A case that goes to the
      // join itself does what the default does, so it needs no test.
      std::vector<BasicBlock*> arms;
      std::vector<Value*> caseVals;
      bool ok = true;
      for (size_t k = 1; k < sw->bbs.size() && ok; ++k) {
        BasicBlock* a = sw->bbs[k];
        if (a == join) continue;
        const Instruction* t = a->insts.back();
        ok = a != h && edgesTo(h, a) == 1 && preds[a].size() == 1 && a->insts.front()->op != Op::Phi &&
             t->op == Op::Br && t->bbs.size() == 1 && t->bbs[0] == join && !done.count(a);
        arms.push_back(a);
        caseVals.push_back(sw->ops[k]);
      }
      if (!ok) continue;
      // Each phi of the join has the same value on every edge from h (default and cases to the join), and one entry
      // per arm.
      std::vector<Instruction*> phis;
      std::vector<Value*> chain;                // P(k): the phi's value when none of the first k cases matched
      std::vector<std::vector<Value*>> armVal;  // armVal[phi][arm]
      for (Instruction* phi : join->insts) {
        if (phi->op != Op::Phi) break;
        Value* v0 = nullptr;
        size_t fromH = 0, fromArms = 0;
        std::vector<Value*> av(arms.size(), nullptr);
        for (size_t k = 0; k < phi->bbs.size(); ++k) {
          if (phi->bbs[k] == h) {
            ok &= !v0 || v0 == phi->ops[k];
            v0 = phi->ops[k];
            ++fromH;
          }
          for (size_t a = 0; a < arms.size(); ++a)
            if (phi->bbs[k] == arms[a]) av[a] = phi->ops[k], ++fromArms;
        }
        ok &= fromH == edgesTo(h, join) && fromArms == arms.size();
        phis.push_back(phi);
        chain.push_back(v0);
        armVal.push_back(std::move(av));
      }
      if (!ok) continue;

      // Drop the entries for h and the arms, then rebuild: h tests the first case, each arm falls into the next
      // test, the last arm falls into the join; the join's phi keeps the last link: [last arm's value], [P(n)].
      for (Instruction* phi : phis)
        for (size_t k = phi->bbs.size(); k-- > 0;)
          if (phi->bbs[k] == h || std::find(arms.begin(), arms.end(), phi->bbs[k]) != arms.end()) eraseEntry(phi, k);
      h->insts.pop_back();
      std::vector<BasicBlock*> tests{h};
      for (size_t k = 1; k < arms.size(); ++k) {
        tests.push_back(m_.addBlock(f_));
        placeAfter[arms[k - 1]] = tests.back();
      }
      Type* i1 = m_.intTy(1);
      for (size_t k = 0; k < arms.size(); ++k) {
        BasicBlock* test = tests[k];
        BasicBlock* next = k + 1 < arms.size() ? tests[k + 1] : join;
        if (k > 0)
          for (size_t p = 0; p < phis.size(); ++p) {
            Instruction* link = m_.newInst(Op::Phi, phis[p]->type);
            link->flags = phis[p]->flags;
            link->ops = {armVal[p][k - 1], chain[p]};
            link->bbs = {arms[k - 1], tests[k - 1]};
            test->insts.push_back(link);
            chain[p] = link;
          }
        Instruction* eq = m_.newInst(Op::Cmp, i1);
        eq->sub = kIntEq;
        eq->ops = {sw->ops[0], caseVals[k]};
        Instruction* br = m_.newInst(Op::Br, sw->type);
        br->ops = {eq};
        br->bbs = {arms[k], next};
        test->insts.push_back(eq);
        test->insts.push_back(br);
        arms[k]->insts.back()->bbs[0] = next;
      }
      if (arms.empty()) {
        Instruction* br = m_.newInst(Op::Br, sw->type);
        br->bbs = {join};
        h->insts.push_back(br);
      }
      for (size_t p = 0; p < phis.size(); ++p) {
        if (!arms.empty()) {
          phis[p]->ops.push_back(armVal[p].back());
          phis[p]->bbs.push_back(arms.back());
        }
        phis[p]->ops.push_back(chain[p]);
        phis[p]->bbs.push_back(arms.empty() ? h : tests.back());
      }
      done.insert(join);
      for (const BasicBlock* a : arms) done.insert(a);
      any = true;
      if (stats_) ++stats_->loweredSwitches;
    }
    if (!any) return false;
    // New test blocks sit right after the arm that falls into them.
    std::unordered_set<const BasicBlock*> placed;
    for (const auto& [arm, test] : placeAfter) placed.insert(test);
    std::vector<BasicBlock*> order;
    for (BasicBlock* b : f_->blocks) {
      if (placed.count(b)) continue;
      order.push_back(b);
      for (auto it = placeAfter.find(b); it != placeAfter.end(); it = placeAfter.find(it->second))
        order.push_back(it->second);
    }
    f_->blocks = std::move(order);
    return true;
  }

  // A
  bool forwardEmpty() {
    Preds preds = predecessors();
    std::unordered_set<const BasicBlock*> touched, removed;
    for (BasicBlock* e : f_->blocks) {
      if (e == f_->blocks[0] || e->insts.size() != 1 || touched.count(e)) continue;
      const Instruction* t = e->insts[0];
      if (t->op != Op::Br || t->bbs.size() != 1) continue;
      BasicBlock* m = t->bbs[0];
      if (m == e || touched.count(m)) continue;
      const std::vector<BasicBlock*>& ps = preds[e];
      bool ok = !ps.empty();
      for (const BasicBlock* p : ps)
        ok &= p != e && p != m && !touched.count(p) && edgesTo(p, e) == 1 && edgesTo(p, m) == 0;
      for (const Instruction* phi : m->insts) {
        if (phi->op != Op::Phi) break;
        size_t at;
        ok &= entryFor(phi, e, at);
      }
      if (!ok) continue;
      for (Instruction* phi : m->insts) {
        if (phi->op != Op::Phi) break;
        size_t at;
        entryFor(phi, e, at);
        Value* v = phi->ops[at];
        eraseEntry(phi, at);
        for (BasicBlock* p : ps) {
          phi->ops.push_back(v);
          phi->bbs.push_back(p);
        }
      }
      for (BasicBlock* p : ps) {
        for (BasicBlock*& s : p->insts.back()->bbs)
          if (s == e) s = m;
        touched.insert(p);
      }
      touched.insert(m);
      removed.insert(e);
      if (stats_) ++stats_->forwardedBlocks;
    }
    if (removed.empty()) return false;
    f_->blocks.erase(
        std::remove_if(f_->blocks.begin(), f_->blocks.end(), [&](BasicBlock* b) { return removed.count(b) != 0; }),
        f_->blocks.end());
    return true;
  }

  // The arm `a` of a conditional branch from h: a single-predecessor block ending in `br join` whose instructions can
  // all run in h. Returns the join, or null.
  BasicBlock* arm(const Preds& preds, const BasicBlock* h, const BasicBlock* a) const {
    if (a == h || edgesTo(h, a) != 1) return nullptr;
    auto p = preds.find(a);
    if (p == preds.end() || p->second.size() != 1) return nullptr;
    const Instruction* t = a->insts.back();
    if (t->op != Op::Br || t->bbs.size() != 1 || t->bbs[0] == a || t->bbs[0] == h) return nullptr;
    for (size_t k = 0; k + 1 < a->insts.size(); ++k)
      if (!speculatable(a->insts[k])) return nullptr;
    return t->bbs[0];
  }

  // B
  bool ifConvert() {
    Preds preds = predecessors();
    std::unordered_set<const BasicBlock*> removed;
    for (BasicBlock* h : f_->blocks) {
      if (removed.count(h)) continue;
      Instruction* br = h->insts.back();
      if (br->op != Op::Br || br->bbs.size() != 2 || br->bbs[0] == br->bbs[1]) continue;
      BasicBlock* x = br->bbs[0];
      BasicBlock* y = br->bbs[1];
      // The arms (null for a side that branches straight to the join) and the join.
      BasicBlock *ax = nullptr, *ay = nullptr, *join = nullptr;
      BasicBlock* jx = removed.count(x) ? nullptr : arm(preds, h, x);
      BasicBlock* jy = removed.count(y) ? nullptr : arm(preds, h, y);
      if (jx && jy && jx == jy) {
        ax = x, ay = y, join = jx;
      } else if (jx && jx == y) {
        ax = x, join = y;
      } else if (jy && jy == x) {
        ay = y, join = x;
      } else {
        continue;
      }
      if (join == h || edgesTo(h, join) != (ax && ay ? 0 : 1)) continue;
      size_t cost = (ax ? ax->insts.size() - 1 : 0) + (ay ? ay->insts.size() - 1 : 0);
      if (cost > kMaxSpeculated) continue;
      // Every phi of the join needs one entry for each side, of a type select handles.
      BasicBlock* fromX = ax ? ax : h;  // the block the phis name for the true side
      BasicBlock* fromY = ay ? ay : h;
      bool ok = true;
      for (const Instruction* phi : join->insts) {
        if (phi->op != Op::Phi) break;
        ok &= selectable(phi->type);
        for (const BasicBlock* from : {fromX, fromY}) ok &= std::count(phi->bbs.begin(), phi->bbs.end(), from) == 1;
      }
      if (!ok) continue;

      Value* cond = br->ops[0];
      h->insts.pop_back();
      for (BasicBlock* a : {ax, ay}) {
        if (!a) continue;
        h->insts.insert(h->insts.end(), a->insts.begin(), a->insts.end() - 1);
        removed.insert(a);
        if (stats_) ++stats_->convertedArms;
      }
      for (Instruction* phi : join->insts) {
        if (phi->op != Op::Phi) break;
        size_t kx, ky;
        entryFor(phi, fromX, kx);
        entryFor(phi, fromY, ky);
        Value* v = phi->ops[kx];
        if (v != phi->ops[ky]) {
          Instruction* sel = m_.newInst(Op::Select, phi->type);
          sel->ops = {cond, phi->ops[kx], phi->ops[ky]};
          h->insts.push_back(sel);
          v = sel;
        }
        // The higher index first, so the lower stays valid.
        eraseEntry(phi, std::max(kx, ky));
        eraseEntry(phi, std::min(kx, ky));
        phi->ops.push_back(v);
        phi->bbs.push_back(h);
      }
      Instruction* jump = m_.newInst(Op::Br, br->type);
      jump->bbs = {join};
      h->insts.push_back(jump);
    }
    if (removed.empty()) return false;
    f_->blocks.erase(
        std::remove_if(f_->blocks.begin(), f_->blocks.end(), [&](BasicBlock* b) { return removed.count(b) != 0; }),
        f_->blocks.end());
    return true;
  }

  Module& m_;
  Function* f_;
  SimplifyCfgStats* stats_;
};

}  // namespace

bool simplifyCfg(Module& m, Function* f, SimplifyCfgStats* stats) {
  if (f->blocks.empty()) return false;
  return Simplifier(m, f, stats).run();
}

}  // namespace spirv2air
