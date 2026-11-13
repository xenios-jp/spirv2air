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
// and the host does not write it while the draw runs. Threadgroup memory is never CSE'd.
//
// Private memory (forwardPrivate, vertex functions): an alloca whose address only reaches loads, stores and GEPs or
// bitcasts of it (it never escapes into a call, a phi, a select or memory) can only be read or written by those, so
// the walk tracks what each such alloca holds, as EarlyCSE does without MemorySSA. A load is replaced by the value that
// the last store through the same pointer (the same IR value, which the walk has already CSE'd) wrote, or by an
// earlier load through it, with the same type, when nothing could have written the alloca in between. Every alloca has
// a generation, renewed by any store into it (whatever the pointer, constant or dynamic index) and, for all allocas,
// on entry to a block with more than one predecessor edge (merges and loop headers: another path may have written
// it); a block with one predecessor starts from its predecessor's state at its end, since it is that block's child in
// the dominator tree. A value is reused only within its generation. In a block, a store through the pointer and type
// of the last store into the same alloca, with no remaining load of that alloca in between, makes the earlier one
// dead, and a store of the value the alloca is known to hold there is dropped. Pixel shaders are left alone: there
// it changed neither Apple's compiled code nor its compile work.
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
bool callHas(const Instruction* i, const Attr& a) {
  if (i->ops.empty() || i->ops[0]->vk != Value::Func) return false;
  return hasFunctionAttr(static_cast<const Function*>(i->ops[0])->attrs, a) || hasFunctionAttr(i->attrs, a);
}

unsigned addrSpaceOf(const Value* ptr) { return ptr->type->kind == Type::Pointer ? ptr->type->n : ~0u; }

// The table compares instructions by (op, sub, flags, type, operands, ...). An instruction's operands are resolved
// before it enters the table and are not changed while it is there, so the table holds the instructions themselves.
struct InstHash {
  size_t operator()(const Instruction* i) const {
    size_t h = size_t(i->op) | size_t(i->sub) << 8 | size_t(i->flags) << 16 | size_t(i->inbounds) << 24 |
               size_t(i->tail) << 25 | size_t(i->align) << 32;
    auto mix = [&h](const void* p) { h = (h ^ reinterpret_cast<size_t>(p)) * 0x9E3779B97F4A7C15ull, h ^= h >> 29; };
    mix(i->type), mix(i->srcElemTy), mix(i->attrs);
    for (const Value* v : i->ops) mix(v);
    for (uint32_t x : i->idx) h = (h ^ x) * 0x9E3779B97F4A7C15ull;
    return h;
  }
};

struct InstEq {
  bool operator()(const Instruction* a, const Instruction* b) const {
    return a->op == b->op && a->sub == b->sub && a->flags == b->flags && a->inbounds == b->inbounds &&
           a->tail == b->tail && a->align == b->align && a->type == b->type && a->srcElemTy == b->srcElemTy &&
           a->attrs == b->attrs && a->ops.size() == b->ops.size() &&
           std::equal(a->ops.begin(), a->ops.end(), b->ops.begin()) && a->idx == b->idx;
  }
};

// A set of instructions, open addressing with linear probing. Entries leave in the reverse order they came in (the
// undo log), so emptying an entry's slot restores the table exactly as it was before the entry went in.
class Table {
 public:
  explicit Table(size_t n) {
    size_t cap = 16;
    while (cap < 2 * n) cap <<= 1;
    slots_.assign(cap, {0, nullptr});
    mask_ = cap - 1;
  }

  // The instruction equal to i already in the table; otherwise inserts i, sets `at` to its slot and returns null.
  Instruction* insert(Instruction* i, size_t& at) {
    const size_t h = InstHash()(i);
    for (size_t k = h & mask_;; k = (k + 1) & mask_) {
      Slot& s = slots_[k];
      if (!s.inst) {
        s = {h, i};
        at = k;
        return nullptr;
      }
      if (s.hash == h && InstEq()(s.inst, i)) return s.inst;
    }
  }

  void erase(size_t at) { slots_[at].inst = nullptr; }

 private:
  struct Slot {
    size_t hash;
    Instruction* inst;
  };
  std::vector<Slot> slots_;
  size_t mask_;
};

// What a load through `ptr` of `type` would read, while the alloca's generation is `gen`.
struct PrivateKey {
  const Value* ptr;
  const Type* type;
  bool operator==(const PrivateKey& o) const { return ptr == o.ptr && type == o.type; }
};
struct PrivateKeyHash {
  size_t operator()(const PrivateKey& k) const {
    size_t h = reinterpret_cast<size_t>(k.ptr) * 0x9E3779B97F4A7C15ull;
    return (h ^ (h >> 29) ^ reinterpret_cast<size_t>(k.type)) * 0x9E3779B97F4A7C15ull;
  }
};
struct PrivateValue {
  Value* value;
  uint32_t gen;
};

class Cse {
 public:
  Cse(Module& m, Function* f, bool forwardPrivate) : f_(f), forwardPrivate_(forwardPrivate), table_(count(f)) {
    ibase_ = m.allocTags(unsigned(count(f)));
    n_ = 0;
    for (BasicBlock* b : f->blocks)
      for (Instruction* i : b->insts) i->tag = ibase_ + n_++;
    repl_.assign(n_, nullptr);
  }

  void run() {
    findWrites();
    if (forwardPrivate_) findPrivate();
    buildTree();
    // Preorder walk with an explicit stack: (block, undo-log sizes on entry, next child).
    std::vector<std::tuple<int, size_t, size_t, size_t>> stack;
    stack.emplace_back(0, 0, 0, 0);
    enter(0, -1);
    visit(0);
    while (!stack.empty()) {
      auto& [b, mark, privateMark, k] = stack.back();
      if (k < children_[b].size()) {
        const int parent = b, c = children_[b][k++];
        stack.emplace_back(c, undo_.size(), privateUndo_.size(), 0);  // invalidates b and k
        enter(c, parent);
        visit(c);
      } else {
        while (undo_.size() > mark) table_.erase(undo_.back()), undo_.pop_back();
        while (privateUndo_.size() > privateMark) {
          auto& [key, had, old] = privateUndo_.back();
          if (had)
            private_[key] = old;
          else
            private_.erase(key);
          privateUndo_.pop_back();
        }
        stack.pop_back();
      }
    }
    if (!replaced_) return;
    // Uses the walk had not reached yet (phi entries along back edges).
    for (BasicBlock* b : f_->blocks)
      for (Instruction* i : b->insts)
        for (Value*& v : i->ops) v = resolve(v);
  }

 private:
  static size_t count(const Function* f) {
    size_t n = 0;
    for (const BasicBlock* b : f->blocks) n += b->insts.size();
    return n;
  }

  Value* resolve(Value* v) const {
    unsigned k = v->tag - ibase_;
    if (v->vk != Value::Inst || k >= n_ || !repl_[k]) return v;
    return repl_[k];  // replacements are recorded already resolved
  }

  // Whether the function can write device memory: a store to address space 1 or a call that is neither readnone nor
  // readonly (atomics, texture writes, barriers).
  void findWrites() {
    for (const BasicBlock* b : f_->blocks)
      for (const Instruction* i : b->insts) {
        if (i->op == Op::Store && addrSpaceOf(i->ops[1]) == 1) deviceWritten_ = true;
        if (i->op == Op::Call && !callHas(i, readnone_) && !callHas(i, readonly_)) deviceWritten_ = true;
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
        return !i->type->isVoid() && callHas(i, readnone_);
      default:
        return false;
    }
  }

  void visit(int bi) {
    BasicBlock* b = f_->blocks[bi];
    kept_.clear();
    if (slots_) std::fill(lastStore_.begin(), lastStore_.end(), kNoStore);
    for (Instruction* i : b->insts) {
      if (i->op != Op::Phi)
        for (Value*& v : i->ops) v = resolve(v);
      if (slots_ && (i->op == Op::Load || i->op == Op::Store) && visitPrivate(i)) continue;
      if (!eligible(i)) {
        kept_.push_back(i);
        continue;
      }
      size_t at;
      if (Instruction* same = table_.insert(i, at)) {
        repl_[i->tag - ibase_] = same;
        replaced_ = true;
      } else {
        undo_.push_back(at);
        kept_.push_back(i);
      }
    }
    if (slots_) {
      kept_.erase(std::remove(kept_.begin(), kept_.end(), nullptr), kept_.end());  // dead stores
      std::copy(gen_.begin(), gen_.end(), endGen_.begin() + size_t(bi) * slots_);
    }
    b->insts.swap(kept_);  // kept_ keeps the old storage for the next block
  }

  // The allocas private memory forwarding may track: those whose address (the alloca, and GEPs and bitcasts of it,
  // transitively) is only used as the address of a load or a store, or as the base of such a GEP or bitcast.
  void findPrivate() {
    std::vector<char> escapes;
    for (const BasicBlock* b : f_->blocks)
      for (const Instruction* i : b->insts)
        if (i->op == Op::Alloca) slotOf_.emplace(i, unsigned(escapes.size())), escapes.push_back(0);
    if (escapes.empty()) return;
    auto derives = [](const Instruction* i) {
      return i->op == Op::GEP || (i->op == Op::Cast && i->sub == BitCast);
    };
    for (bool changed = true; changed;) {
      changed = false;
      for (const BasicBlock* b : f_->blocks)
        for (const Instruction* i : b->insts)
          if (derives(i) && !slotOf_.count(i))
            if (auto it = slotOf_.find(i->ops[0]); it != slotOf_.end()) slotOf_.emplace(i, it->second), changed = true;
    }
    for (const BasicBlock* b : f_->blocks)
      for (const Instruction* i : b->insts)
        for (size_t k = 0; k < i->ops.size(); ++k) {
          auto it = slotOf_.find(i->ops[k]);
          if (it == slotOf_.end()) continue;
          const bool address = (i->op == Op::Load && k == 0) || (i->op == Op::Store && k == 1) || (derives(i) && k == 0);
          if (!address) escapes[it->second] = 1;
        }
    // Number the tracked allocas densely; the others are dropped from the map.
    std::vector<unsigned> dense(escapes.size(), kNoSlot);
    for (size_t a = 0; a < escapes.size(); ++a)
      if (!escapes[a]) dense[a] = slots_++;
    for (auto it = slotOf_.begin(); it != slotOf_.end();)
      if (dense[it->second] == kNoSlot)
        it = slotOf_.erase(it);
      else
        it->second = dense[it->second], ++it;
    gen_.assign(slots_, 0);
    lastStore_.assign(slots_, kNoStore);
  }

  unsigned slotOf(const Value* ptr) const {
    auto it = slotOf_.find(ptr);
    return it == slotOf_.end() ? kNoSlot : it->second;
  }

  // Block c's private memory state on entry: its dominator-tree parent's at its end, or nothing known after a merge.
  void enter(int c, int parent) {
    if (!slots_) return;
    if (parent >= 0)
      std::copy_n(endGen_.begin() + size_t(parent) * slots_, slots_, gen_.begin());
    if (parent < 0 || predCount_[c] != 1)
      for (uint32_t& g : gen_) g = ++nextGen_;
  }

  void setPrivate(const PrivateKey& key, PrivateValue v) {
    auto [it, inserted] = private_.try_emplace(key, v);
    privateUndo_.push_back({key, !inserted, inserted ? PrivateValue{} : it->second});
    if (!inserted) it->second = v;
  }

  // A load or store of a tracked alloca; false for any other.
  bool visitPrivate(Instruction* i) {
    const bool load = i->op == Op::Load;
    Value* ptr = i->ops[load ? 0 : 1];
    const unsigned s = slotOf(ptr);
    if (s == kNoSlot) return false;
    if (i->isVolatile) {
      if (!load) gen_[s] = ++nextGen_;
      lastStore_[s] = kNoStore;
      kept_.push_back(i);
      return true;
    }
    const PrivateKey key{ptr, load ? i->type : i->ops[0]->type};
    auto it = private_.find(key);
    const bool known = it != private_.end() && it->second.gen == gen_[s];
    if (load) {
      if (known) {
        repl_[i->tag - ibase_] = it->second.value;
        replaced_ = true;
        return true;
      }
      setPrivate(key, {i, gen_[s]});
      lastStore_[s] = kNoStore;
      kept_.push_back(i);
      return true;
    }
    if (known && it->second.value == i->ops[0]) return true;  // the alloca already holds the value
    if (lastStore_[s] != kNoStore) {
      const Instruction* last = kept_[lastStore_[s]];
      if (last->ops[1] == ptr && last->ops[0]->type == i->ops[0]->type) kept_[lastStore_[s]] = nullptr;
    }
    gen_[s] = ++nextGen_;
    setPrivate(key, {i->ops[0], gen_[s]});
    lastStore_[s] = kept_.size();
    kept_.push_back(i);
    return true;
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
    if (slots_) {
      predCount_.assign(n, 0);
      for (size_t b = 0; b < n; ++b) predCount_[b] = unsigned(pred[b].size());
      endGen_.assign(n * slots_, 0);
    }
    children_.assign(n, {});
    // Function order, so the walk (and the output) does not depend on the traversal above.
    for (size_t b = 1; b < n; ++b)
      if (idom[b] >= 0) children_[idom[b]].push_back(int(b));
  }

  static constexpr unsigned kNoSlot = ~0u;
  static constexpr size_t kNoStore = ~size_t(0);

  Function* f_;
  const bool forwardPrivate_;
  // Private memory: tracked alloca number of each alloca and pointer derived from one; per tracked alloca its current
  // generation and, in the current block, the index in kept_ of its last store that no remaining load has read since.
  std::unordered_map<const Value*, unsigned> slotOf_;
  unsigned slots_ = 0;
  uint32_t nextGen_ = 0;
  std::vector<uint32_t> gen_, endGen_;  // endGen_: by block, each block's gen_ at its end
  std::vector<unsigned> predCount_;     // predecessor edges of each block
  std::vector<size_t> lastStore_;
  std::unordered_map<PrivateKey, PrivateValue, PrivateKeyHash> private_;
  struct PrivateUndo {
    PrivateKey key;
    bool had;
    PrivateValue old;
  };
  std::vector<PrivateUndo> privateUndo_;
  const Attr readnone_ = Attr::get("readnone"), readonly_ = Attr::get("readonly");
  bool deviceWritten_ = false, replaced_ = false;
  std::vector<std::vector<int>> children_;
  Table table_;
  std::vector<size_t> undo_;  // the slots of the table's entries in insertion order, emptied when their block's
                              // subtree ends
  std::vector<Instruction*> kept_;
  uint32_t ibase_ = 0;
  unsigned n_ = 0;
  std::vector<Value*> repl_;  // by instruction tag: the available instruction that replaces it
};

}  // namespace

void cseFunction(Module& m, Function* f, bool forwardPrivate) {
  if (f->blocks.empty()) return;
  Cse(m, f, forwardPrivate).run();
}

}  // namespace spirv2air
