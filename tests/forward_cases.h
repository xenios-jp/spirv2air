// Private memory forwarding in cseFunction (forward_cpu, forward_gpu): each case is a compute kernel that stores to and
// loads from allocas in a pattern where a careless forward would read the wrong value (aliasing through a dynamic
// index, a store in one branch, a loop back edge, a sibling branch's store, an alloca whose address escapes, a store
// that an earlier load still needs), with the number of private loads and stores that must remain after the pass.
// forward_cpu checks those counts; forward_gpu runs the kernel with and without the pass on every row of operands and
// compares both with the host's value of the same program.
//
// Every thread reads one row of operands, in[6 * gid ...] = (i, j, c, k, x, y): i, j are indices 0..3 into the
// alloca A (a [4 x i32] that each kernel first fills with zeros), c a branch condition, k a loop count, x and y values.
// It writes out[2 * gid], out[2 * gid + 1].
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "airkit/airkit.h"
#include "cse.h"

namespace forwardcases {

using airbc::BasicBlock;
using airbc::Builder;
using airbc::Instruction;
using airbc::Value;

struct Row {
  uint32_t i, j, c, k, x, y;
};

struct Ctx {
  airkit::Kernel& k;
  Builder& b;
  Value *i, *j, *c, *n, *x, *y;  // c is an i1
  Value* a;                      // the [4 x i32] alloca
  Value* zero;

  Value* el(Value* index) {
    return b.gep(k.m.arrayTy(k.i32, 4), a, {k.m.constInt(k.i32, 0), index}, true);
  }
  Value* el(int index) { return el(k.m.constInt(k.i32, index)); }
  Value* load(Value* p) { return b.load(k.i32, p, 4); }
  void store(Value* v, Value* p) { b.store(v, p, 4); }
  BasicBlock* block(const char* name) { return k.block(name); }
};

struct Case {
  std::string name;
  unsigned loads, stores;  // private loads and stores that remain after forwarding (the zero fill is 4 stores)
  // Emits the body after A's zero fill; returns the two results (in the block the builder ends in).
  std::function<std::pair<Value*, Value*>(Ctx&)> build;
  std::function<std::pair<uint32_t, uint32_t>(const Row&)> want;
};

inline std::vector<Case> cases() {
  return {
      // A load through the pointer of the last store takes its value; one through another pointer does not.
      {"store-load", 1, 5,
       [](Ctx& t) {
         t.store(t.x, t.el(0));
         return std::make_pair(t.load(t.el(0)), t.load(t.el(t.i)));
       },
       [](const Row& r) { return std::make_pair(r.x, r.i == 0 ? r.x : 0u); }},
      // A[i] = x; A[j] = y; A[i] is y when i == j: the second store renews everything known about A.
      {"dynamic-alias", 1, 6,
       [](Ctx& t) {
         t.store(t.x, t.el(t.i));
         t.store(t.y, t.el(t.j));
         return std::make_pair(t.load(t.el(t.i)), t.load(t.el(t.j)));
       },
       [](const Row& r) { return std::make_pair(r.i == r.j ? r.y : r.x, r.y); }},
      // A store in one arm: the merge block knows nothing; its second load reuses its first.
      {"branch-store", 1, 6,
       [](Ctx& t) {
         t.store(t.x, t.el(0));
         BasicBlock *then = t.block("then"), *merge = t.block("merge");
         t.b.condBr(t.c, then, merge);
         t.b.setInsertBlock(then);
         t.store(t.y, t.el(0));
         t.b.br(merge);
         t.b.setInsertBlock(merge);
         return std::make_pair(t.load(t.el(0)), t.load(t.el(0)));
       },
       [](const Row& r) { return std::make_pair(r.c ? r.y : r.x, r.c ? r.y : r.x); }},
      // A loop that increments A[0] k times: the header's load must not take the value stored before the loop.
      {"loop", 1, 6,
       [](Ctx& t) {
         t.store(t.x, t.el(0));
         BasicBlock* entry = t.k.function()->blocks.back();
         BasicBlock *head = t.block("head"), *body = t.block("body"), *exit = t.block("exit");
         t.b.br(head);
         t.b.setInsertBlock(head);
         Instruction* count = t.b.phi(t.k.i32, {{t.zero, entry}});
         Value* v = t.load(t.el(0));
         t.b.condBr(t.b.icmp(36 /* ult */, count, t.n), body, exit);
         t.b.setInsertBlock(body);
         t.store(t.b.binop(airbc::Add, v, t.k.m.constInt(t.k.i32, 1)), t.el(0));
         Value* next = t.b.binop(airbc::Add, count, t.k.m.constInt(t.k.i32, 1));
         count->ops.push_back(next);
         count->bbs.push_back(body);
         t.b.br(head);
         t.b.setInsertBlock(exit);
         return std::make_pair(v, t.load(t.el(0)));  // exit's one predecessor is head: its load reuses v
       },
       [](const Row& r) { return std::make_pair(r.x + r.k, r.x + r.k); }},
      // Sibling arms: the else arm must see the value from before the branch, not the then arm's store.
      {"sibling", 1, 6,
       [](Ctx& t) {
         t.store(t.x, t.el(0));
         BasicBlock *then = t.block("then"), *other = t.block("else"), *merge = t.block("merge");
         t.b.condBr(t.c, then, other);
         t.b.setInsertBlock(then);
         t.store(t.y, t.el(0));
         Value* a = t.load(t.el(0));
         t.b.br(merge);
         t.b.setInsertBlock(other);
         Value* b = t.load(t.el(0));
         t.b.br(merge);
         t.b.setInsertBlock(merge);
         Value* p = t.b.phi(t.k.i32, {{a, then}, {b, other}});
         return std::make_pair(p, t.load(t.el(1)));
       },
       [](const Row& r) { return std::make_pair(r.c ? r.y : r.x, 0u); }},
      // A store read by a load through another pointer before it is overwritten stays.
      {"store-read-overwrite", 1, 6,
       [](Ctx& t) {
         t.store(t.x, t.el(t.i));
         Value* a = t.load(t.el(0));
         t.store(t.y, t.el(t.i));
         return std::make_pair(a, t.load(t.el(t.i)));
       },
       [](const Row& r) { return std::make_pair(r.i == 0 ? r.x : 0u, r.y); }},
      // A store overwritten through the same pointer with nothing reading A in between is dead.
      {"overwrite", 1, 5,
       [](Ctx& t) {
         t.store(t.x, t.el(t.i));
         t.store(t.y, t.el(t.i));
         return std::make_pair(t.load(t.el(t.i)), t.load(t.el(t.j)));
       },
       [](const Row& r) { return std::make_pair(r.y, r.i == r.j ? r.y : 0u); }},
      // A whole-vector store, then one component through a GEP: the whole load must read memory.
      {"vector-component", 1, 6,
       [](Ctx& t) {
         airbc::Type* v4 = t.k.m.vecTy(t.k.i32, 4);
         Value* v = t.b.allocaInst(v4, 16);
         Value* splat = t.k.m.constUndef(v4);
         for (int e = 0; e < 4; ++e) splat = t.b.insertElement(splat, t.x, t.k.m.constInt(t.k.i32, e));
         t.b.store(splat, v, 16);
         t.store(t.y, t.b.gep(v4, v, {t.k.m.constInt(t.k.i32, 0), t.k.m.constInt(t.k.i32, 1)}, true));
         Value* w = t.b.load(v4, v, 16);
         return std::make_pair(t.b.extractElement(w, t.k.m.constInt(t.k.i32, 1)),
                               t.b.extractElement(w, t.k.m.constInt(t.k.i32, 0)));
       },
       [](const Row& r) { return std::make_pair(r.y, r.x); }},
      // Two allocas whose addresses meet in a select: neither is tracked.
      {"escape-select", 2, 7,
       [](Ctx& t) {
         Value* p = t.b.allocaInst(t.k.i32, 4);
         Value* q = t.b.allocaInst(t.k.i32, 4);
         t.store(t.x, p);
         t.store(t.zero, q);
         t.store(t.y, t.b.select(t.c, p, q));
         return std::make_pair(t.load(p), t.load(q));
       },
       [](const Row& r) { return std::make_pair(r.c ? r.y : r.x, r.c ? 0u : r.y); }},
      // Load, store through another index, load again: the second load is not the first when i == j.
      {"load-store-load", 2, 5,
       [](Ctx& t) {
         Value* a = t.load(t.el(t.i));
         t.store(t.y, t.el(t.j));
         return std::make_pair(a, t.load(t.el(t.i)));
       },
       [](const Row& r) { return std::make_pair(0u, r.i == r.j ? r.y : 0u); }},
      // Storing back an earlier load's value after another store is not a no-op; storing it twice is.
      {"store-back", 1, 7,
       [](Ctx& t) {
         t.store(t.x, t.el(t.i));
         Value* a = t.load(t.el(t.i));
         t.store(t.y, t.el(t.j));
         t.store(a, t.el(t.i));
         t.store(a, t.el(t.i));
         return std::make_pair(t.load(t.el(t.i)), t.load(t.el(t.j)));
       },
       [](const Row& r) { return std::make_pair(r.x, r.i == r.j ? r.x : r.y); }},
      // One-predecessor arms continue from the branch's state; the merge does not.
      {"one-predecessor", 2, 6,
       [](Ctx& t) {
         t.store(t.x, t.el(t.i));
         BasicBlock *then = t.block("then"), *other = t.block("else"), *merge = t.block("merge");
         t.b.condBr(t.c, then, other);
         t.b.setInsertBlock(then);
         Value* a = t.load(t.el(t.i));
         t.b.br(merge);
         t.b.setInsertBlock(other);
         t.store(t.y, t.el(t.j));
         Value* b = t.load(t.el(t.i));
         t.b.br(merge);
         t.b.setInsertBlock(merge);
         Value* p = t.b.phi(t.k.i32, {{a, then}, {b, other}});
         return std::make_pair(p, t.load(t.el(t.j)));
       },
       [](const Row& r) {
         uint32_t a[4] = {0, 0, 0, 0};
         a[r.i] = r.x;
         if (!r.c) a[r.j] = r.y;
         return std::make_pair(a[r.i], a[r.j]);
       }},
  };
}

// The kernel of case c: reads its row, fills A with zeros, runs the case, writes the results.
inline void build(airkit::Kernel& k, const Case& c) {
  unsigned in = k.buffer(k.i32, 0, airkit::Access::Read, "in");
  unsigned out = k.buffer(k.i32, 1, airkit::Access::ReadWrite, "out");
  unsigned gid = k.builtin(airkit::Builtin::ThreadPositionInGrid);
  k.begin();
  Builder& b = k.builder();
  Value* g = b.cast(airbc::ZExt, k.arg(gid), k.i64);
  auto operand = [&](int n) {
    Value* idx = b.binop(airbc::Add, b.binop(airbc::Mul, g, k.m.constInt(k.i64, 6)), k.m.constInt(k.i64, n));
    return b.load(k.i32, b.gep(k.i32, k.arg(in), {idx}, true), 4);
  };
  Value* zero = k.m.constInt(k.i32, 0);
  Value* a = b.allocaInst(k.m.arrayTy(k.i32, 4), 4);
  Ctx t{k, b, operand(0), operand(1), nullptr, operand(3), operand(4), operand(5), a, zero};
  t.c = b.icmp(33 /* ne */, operand(2), zero);
  for (int e = 0; e < 4; ++e) t.store(zero, t.el(e));
  auto [r0, r1] = c.build(t);
  Value* o = b.binop(airbc::Mul, g, k.m.constInt(k.i64, 2));
  b.store(r0, b.gep(k.i32, k.arg(out), {o}, true), 4);
  b.store(r1, b.gep(k.i32, k.arg(out), {b.binop(airbc::Add, o, k.m.constInt(k.i64, 1))}, true), 4);
  b.retVoid();
}

// Loads and stores of private memory (address space 0) left in the kernel.
inline std::pair<unsigned, unsigned> privateAccesses(const airbc::Function* f) {
  unsigned loads = 0, stores = 0;
  for (const BasicBlock* bb : f->blocks)
    for (const Instruction* i : bb->insts) {
      if (i->op == airbc::Op::Load && i->ops[0]->type->n == 0) ++loads;
      if (i->op == airbc::Op::Store && i->ops[1]->type->n == 0) ++stores;
    }
  return {loads, stores};
}

inline std::vector<Row> rows() {
  std::vector<Row> r;
  for (uint32_t i = 0; i < 4; ++i)
    for (uint32_t j = 0; j < 4; ++j)
      for (uint32_t c = 0; c < 2; ++c)
        for (uint32_t k : {0u, 1u, 3u}) r.push_back({i, j, c, k, 100 + uint32_t(r.size()), 7000 + uint32_t(r.size())});
  return r;
}

}  // namespace forwardcases
