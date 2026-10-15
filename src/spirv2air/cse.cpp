// D3 CSE: one preorder walk of the dominator tree with a scoped hash table of the available instructions. An
// instruction whose (op, sub, flags, type, operands, ...) equal those of an available one is replaced by it; the scope
// of an entry ends with its block's subtree, so the available instruction always dominates the replaced one. Dominators
// by Cooper, Harvey and Kennedy, as in mem2reg.cpp. Blocks and instructions are visited in function order, so the
// output does not depend on the hash table.
//
// Loads. Address space 2 is constant memory (the top-level argument buffer, constant buffers, the descriptor heaps'
// entries, which are loaded through pointers into it): AIR does not let a shader write it, so two loads of one pointer
// give one value wherever they are. Address space 1 (device memory) holds the stage's texture-index buffer and the
// shared-memory buffers; it is treated as read-only only in a function that has no store to address space 1 and no call
// that may write memory (everything but readnone and readonly callees), since then nothing in the shader can change it
// and the host does not write it while the draw runs. Private (alloca) and threadgroup memory are never CSE'd.
#include "spirv2air/cse.h"

#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "airbc.h"

namespace spirv2air {

using namespace airbc;

namespace {

bool hasFunctionAttr(const AttrList* l, const Attr& a) {
  if (!l) return false;
  for (const auto& [index, set] : l->sets)
    if (index == AttrList::FunctionIndex && std::find(set->begin(), set->end(), a) != set->end()) return true;
  return false;
}

// A call whose callee (or call site) carries the given function attribute.
bool callHas(const Instruction* i, const char* name) {
  const Attr a = Attr::get(name);
  if (i->ops.empty() || i->ops[0]->vk != Value::Func) return false;
  return hasFunctionAttr(static_cast<const Function*>(i->ops[0])->attrs, a) || hasFunctionAttr(i->attrs, a);
}

unsigned addrSpaceOf(const Value* ptr) { return ptr->type->kind == Type::Pointer ? ptr->type->n : ~0u; }

struct Key {
  Op op;
  uint8_t sub, flags;
  bool inbounds, tail;
  uint32_t align;
  const Type* type;
  const Type* srcElemTy;
  const AttrList* attrs;
  std::vector<const Value*> ops;
  std::vector<uint32_t> idx;

  bool operator==(const Key& o) const {
    return op == o.op && sub == o.sub && flags == o.flags && inbounds == o.inbounds && tail == o.tail &&
           align == o.align && type == o.type && srcElemTy == o.srcElemTy && attrs == o.attrs && ops == o.ops &&
           idx == o.idx;
  }
};

struct KeyHash {
  size_t operator()(const Key& k) const {
    size_t h = size_t(k.op) | size_t(k.sub) << 8 | size_t(k.flags) << 16 | size_t(k.inbounds) << 24 |
               size_t(k.tail) << 25 | size_t(k.align) << 32;
    auto mix = [&h](const void* p) { h = (h ^ reinterpret_cast<size_t>(p)) * 0x9E3779B97F4A7C15ull, h ^= h >> 29; };
    mix(k.type), mix(k.srcElemTy), mix(k.attrs);
    for (const Value* v : k.ops) mix(v);
    for (uint32_t x : k.idx) h = (h ^ x) * 0x9E3779B97F4A7C15ull;
    return h;
  }
};

class Cse {
 public:
  explicit Cse(Function* f) : f_(f) {}

  void run() {
    findWrites();
    buildTree();
    // Preorder walk with an explicit stack: (block, undo-log size on entry, next child).
    std::vector<std::tuple<int, size_t, size_t>> stack;
    stack.emplace_back(0, 0, 0);
    visit(0);
    while (!stack.empty()) {
      auto& [b, mark, k] = stack.back();
      if (k < children_[b].size()) {
        int c = children_[b][k++];
        stack.emplace_back(c, undo_.size(), 0);
        visit(c);
      } else {
        while (undo_.size() > mark) table_.erase(undo_.back()), undo_.pop_back();
        stack.pop_back();
      }
    }
    if (repl_.empty()) return;
    // Uses the walk had not reached yet (phi entries along back edges).
    for (BasicBlock* b : f_->blocks)
      for (Instruction* i : b->insts)
        for (Value*& v : i->ops) v = resolve(v);
  }

 private:
  Value* resolve(Value* v) const {
    auto it = repl_.find(v);
    return it == repl_.end() ? v : it->second;  // replacements are recorded already resolved
  }

  // Whether the function can write device memory: a store to address space 1 or a call that is neither readnone nor
  // readonly (atomics, texture writes, barriers).
  void findWrites() {
    for (const BasicBlock* b : f_->blocks)
      for (const Instruction* i : b->insts) {
        if (i->op == Op::Store && addrSpaceOf(i->ops[1]) == 1) deviceWritten_ = true;
        if (i->op == Op::Call && !callHas(i, "readnone") && !callHas(i, "readonly")) deviceWritten_ = true;
      }
  }

  bool eligible(const Instruction* i) const {
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
        return true;
      case Op::Load: {
        if (i->isVolatile) return false;
        unsigned as = addrSpaceOf(i->ops[0]);
        return as == 2 || (as == 1 && !deviceWritten_);
      }
      case Op::Call:
        return !i->type->isVoid() && callHas(i, "readnone");
      default:
        return false;
    }
  }

  void visit(int bi) {
    BasicBlock* b = f_->blocks[bi];
    std::vector<Instruction*> kept;
    kept.reserve(b->insts.size());
    for (Instruction* i : b->insts) {
      if (i->op != Op::Phi)
        for (Value*& v : i->ops) v = resolve(v);
      if (!eligible(i)) {
        kept.push_back(i);
        continue;
      }
      Key k{i->op, i->sub, i->flags, i->inbounds, i->tail, i->align, i->type, i->srcElemTy, i->attrs, {}, i->idx};
      k.ops.assign(i->ops.begin(), i->ops.end());
      auto [it, inserted] = table_.emplace(std::move(k), i);
      if (inserted) {
        undo_.push_back(it->first);
        kept.push_back(i);
      } else {
        repl_.emplace(i, it->second);
      }
    }
    b->insts = std::move(kept);
  }

  // Dominator tree of the blocks reachable from the entry (iterative CHK); children_ in function order.
  void buildTree() {
    const auto& blocks = f_->blocks;
    const size_t n = blocks.size();
    std::unordered_map<const BasicBlock*, int> id;
    for (size_t b = 0; b < n; ++b) id.emplace(blocks[b], int(b));
    std::vector<std::vector<int>> succ(n), pred(n);
    for (size_t b = 0; b < n; ++b) {
      if (blocks[b]->insts.empty()) continue;
      const Instruction* t = blocks[b]->insts.back();
      if (t->op != Op::Br && t->op != Op::Switch) continue;
      for (BasicBlock* s : t->bbs) succ[b].push_back(id.at(s)), pred[id.at(s)].push_back(int(b));
    }
    std::vector<int> post, po(n, -1);
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
    children_.assign(n, {});
    // Function order, so the walk (and the output) does not depend on the traversal above.
    for (size_t b = 1; b < n; ++b)
      if (idom[b] >= 0) children_[idom[b]].push_back(int(b));
  }

  Function* f_;
  bool deviceWritten_ = false;
  std::vector<std::vector<int>> children_;
  std::unordered_map<Key, Instruction*, KeyHash> table_;
  std::vector<Key> undo_;
  std::unordered_map<Value*, Value*> repl_;
};

}  // namespace

void cseFunction(Module&, Function* f) {
  if (f->blocks.empty()) return;
  Cse(f).run();
}

}  // namespace spirv2air
