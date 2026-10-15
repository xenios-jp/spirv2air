// airbc: a self-contained LLVM-13-layout bitcode writer for AIR (typed pointers).
//
// Scope: the subset of LLVM IR that AIR compute/graphics shaders use. The module model is deliberately small;
// the writer mirrors LLVM 13's ValueEnumerator + ModuleBitcodeWriter so that a module built in the same order
// as `metal-as` parses it serializes to the same bytes.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace airbc {

// ---------------------------------------------------------------- types
struct Constant;
struct Type {
  enum Kind : uint8_t { Void, Half, Float, Double, Label, Metadata, Int, Pointer, Vector, Array, Struct, Function };
  Kind kind;
  uint32_t n = 0;            // Int: width; Vector/Array: count; Pointer: address space
  Type *elem = nullptr;      // Pointer: pointee; Vector/Array: element; Function: return type
  std::vector<Type *> elems; // Struct: elements; Function: params
  std::string name;          // named struct (empty for literal)
  bool packed = false, opaque = false, vararg = false, literal = false;
  mutable uint32_t tag = 0; // the writer's cache of this type's id + 1; checked against the id list before use
  Constant *nullC = nullptr, *undefC = nullptr, *poisonC = nullptr; // Module's interned constants of this type

  bool isVoid() const { return kind == Void; }
  bool isIntOrIntVector() const { return kind == Int || (kind == Vector && elem->kind == Int); }
};

// ---------------------------------------------------------------- attributes
struct Attr {
  enum Cls : uint8_t { Enum = 0, Int = 1, Str = 3, StrVal = 4 };
  Cls cls = Enum;
  uint32_t code = 0; // bitcode ATTR_KIND_*
  int rank = 0;      // LLVM 13 Attribute::AttrKind order (enum/int); strings sort after
  uint64_t value = 0;
  std::string key, val;
  static Attr get(const char *name, uint64_t intValue = 0); // "nounwind", "readnone", "align" (with value) ...
  static Attr str(std::string k, std::string v = "");
  bool operator<(const Attr &o) const;
  bool operator==(const Attr &o) const {
    return cls == o.cls && code == o.code && value == o.value && key == o.key && val == o.val;
  }
};
using AttrSet = std::vector<Attr>; // canonical (sorted, unique)
struct AttrList {
  static constexpr uint32_t FunctionIndex = ~0u, ReturnIndex = 0; // params are 1 + i
  std::vector<std::pair<uint32_t, const AttrSet *>> sets;         // in LLVM index order: fn, ret, params
};

// ---------------------------------------------------------------- values
// The part of std::vector's interface the IR uses, for trivially copyable T, with room for N elements inline: most
// instructions have a handful of operands, and this saves their heap allocation.
template <class T, unsigned N> class SmallVec {
  static_assert(std::is_trivially_copyable<T>::value, "SmallVec holds trivially copyable elements");

public:
  using value_type = T;
  using iterator = T *;
  using const_iterator = const T *;
  SmallVec() = default;
  SmallVec(std::initializer_list<T> l) { assign(l.begin(), l.end()); }
  SmallVec(const SmallVec &o) { assign(o.begin(), o.end()); }
  SmallVec(SmallVec &&o) noexcept { steal(o); }
  ~SmallVec() { std::free(heap_); }
  SmallVec &operator=(const SmallVec &o) {
    if (this != &o) assign(o.begin(), o.end());
    return *this;
  }
  SmallVec &operator=(SmallVec &&o) noexcept {
    if (this != &o) {
      std::free(heap_);
      steal(o);
    }
    return *this;
  }
  SmallVec &operator=(std::initializer_list<T> l) {
    assign(l.begin(), l.end());
    return *this;
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T *data() { return heap_ ? heap_ : inl_; }
  const T *data() const { return heap_ ? heap_ : inl_; }
  T *begin() { return data(); }
  T *end() { return data() + size_; }
  const T *begin() const { return data(); }
  const T *end() const { return data() + size_; }
  T &operator[](size_t i) { return data()[i]; }
  const T &operator[](size_t i) const { return data()[i]; }
  T &front() { return data()[0]; }
  T &back() { return data()[size_ - 1]; }
  const T &front() const { return data()[0]; }
  const T &back() const { return data()[size_ - 1]; }
  void clear() { size_ = 0; }
  void reserve(size_t n) {
    if (n > cap_) grow(n);
  }
  void push_back(const T &v) {
    if (size_ == cap_) {
      T copy = v; // v may point into our storage
      grow(size_ + 1);
      data()[size_++] = copy;
    } else {
      data()[size_++] = v;
    }
  }
  void pop_back() { --size_; }
  void resize(size_t n) {
    reserve(n);
    for (size_t k = size_; k < n; ++k) data()[k] = T();
    size_ = uint32_t(n);
  }
  template <class It> void assign(It first, It last) {
    size_t n = size_t(last - first);
    size_ = 0;
    reserve(n);
    T *d = data();
    for (size_t k = 0; first != last; ++first, ++k) d[k] = *first;
    size_ = uint32_t(n);
  }
  template <class It> T *insert(const T *pos, It first, It last) {
    size_t at = size_t(pos - data()), n = size_t(last - first);
    reserve(size_ + n);
    T *d = data();
    std::memmove(d + at + n, d + at, (size_ - at) * sizeof(T));
    for (size_t k = 0; first != last; ++first, ++k) d[at + k] = *first;
    size_ += uint32_t(n);
    return d + at;
  }
  T *insert(const T *pos, const T &v) { return insert(pos, &v, &v + 1); }
  T *erase(const T *first, const T *last) {
    size_t at = size_t(first - data()), n = size_t(last - first);
    T *d = data();
    std::memmove(d + at, d + at + n, (size_ - at - n) * sizeof(T));
    size_ -= uint32_t(n);
    return d + at;
  }
  T *erase(const T *pos) { return erase(pos, pos + 1); }

private:
  void grow(size_t n) {
    size_t cap = cap_ * 2 > n ? cap_ * 2 : n;
    T *h = static_cast<T *>(std::malloc(cap * sizeof(T)));
    std::memcpy(h, data(), size_ * sizeof(T));
    std::free(heap_);
    heap_ = h;
    cap_ = uint32_t(cap);
  }
  void steal(SmallVec &o) {
    size_ = o.size_;
    if (o.heap_) {
      heap_ = o.heap_;
      cap_ = o.cap_;
      o.heap_ = nullptr;
      o.cap_ = N;
    } else {
      heap_ = nullptr;
      cap_ = N;
      std::memcpy(inl_, o.inl_, size_ * sizeof(T));
    }
    o.size_ = 0;
  }
  T *heap_ = nullptr;
  uint32_t size_ = 0, cap_ = N;
  T inl_[N];
};

struct Value {
  enum VKind : uint8_t { Func, Arg, Const, Inst, Block, Global } vk;
  mutable uint32_t tag = 0; // scratch for passes (dense ids, see Module::allocTags); the writer caches ids + 1 here
  Type *type;
  std::string name;
  Value(VKind k, Type *t) : vk(k), type(t) {}
  virtual ~Value() = default;
};

struct Constant : Value {
  enum CKind : uint8_t { Int, FP, Null, Undef, Poison, Aggregate, Expr, GlobalVar } ck;
  uint64_t bits = 0; // Int: value truncated to width; FP: IEEE bit pattern
  std::vector<Constant *> ops;
  // Expr (ConstantExpr): which instruction it folds, plus its operand encoding.
  enum ExprOp : uint8_t { CE_Cast, CE_GEP, CE_BinOp } eop = CE_Cast;
  uint8_t sub = 0;             // CastOpc / BinOpc
  uint8_t flags = 0;           // BinOp wrap/exact flags
  bool inbounds = false;       // GEP
  Type *srcElemTy = nullptr;   // GEP source element type
  Constant(Type *t, CKind k) : Value(Const, t), ck(k) {}
};

// A module-level global variable (e.g. threadgroup memory: addrspace(3), internal, unnamed_addr, undef).
// Its Value type is the pointer type; valueType is the pointee.
struct GlobalVariable : Constant {
  Type *valueType;
  Constant *init = nullptr;    // nullptr => declaration
  uint32_t align = 0;          // bytes
  uint32_t linkage = 0;        // bitcode linkage encoding: 0 external, 3 internal, 9 private
  uint32_t unnamedAddr = 0;    // 1 unnamed_addr, 2 local_unnamed_addr
  bool isConstant = false;
  bool externallyInitialized = false;
  std::string section;  // empty: none
  GlobalVariable(Type *ptrTy, Type *vt) : Constant(ptrTy, GlobalVar), valueType(vt) { vk = Global; }
};

struct Argument : Value {
  unsigned index;
  Argument(Type *t, unsigned i) : Value(Arg, t), index(i) {}
};

struct BasicBlock;
struct Function;

enum class Op : uint8_t {
  Ret, Br, Unreachable, BinOp, Cast, GEP, Load, Store, ExtractValue, InsertValue,
  ExtractElement, InsertElement, ShuffleVector, Cmp, Phi, Select, Call, Alloca, Switch, FNeg,
};
// Bitcode encodings (LLVM 13 LLVMBitCodes.h)
enum BinOpc : uint8_t { Add = 0, Sub = 1, Mul = 2, UDiv = 3, SDiv = 4, URem = 5, SRem = 6, Shl = 7, LShr = 8,
                        AShr = 9, And = 10, Or = 11, Xor = 12, FAdd = Add, FSub = Sub, FMul = Mul, FDiv = SDiv,
                        FRem = SRem };
enum CastOpc : uint8_t { Trunc = 0, ZExt, SExt, FPToUI, FPToSI, UIToFP, SIToFP, FPTrunc, FPExt, PtrToInt,
                         IntToPtr, BitCast, AddrSpaceCast };
// Fast-math flags, bitcode encoding (bit 0 is the pre-LLVM-6 'unsafe-algebra' bit and is never written).
enum : uint8_t {
  FMF_Reassoc = 1 << 7, FMF_NNaN = 1 << 1, FMF_NInf = 1 << 2, FMF_NSZ = 1 << 3, FMF_ARcp = 1 << 4,
  FMF_Contract = 1 << 5, FMF_Afn = 1 << 6, FMF_Fast = 0xFE, FMF_Strict = 0,
  OBO_NUW = 1, OBO_NSW = 2, PEO_Exact = 1
};
// Linkage, bitcode encoding.
enum : uint32_t { Link_External = 0, Link_Internal = 3, Link_Private = 9 };

struct Instruction : Value {
  Op op;
  uint8_t sub = 0;   // BinOpc / CastOpc / cmp predicate
  uint8_t flags = 0; // wrap / exact / fast-math flags (bitcode encoding)
  bool inbounds = false, isVolatile = false, tail = false;
  uint32_t align = 0;          // bytes
  Type *srcElemTy = nullptr;   // GEP source element type; Call: function type; Alloca: allocated type
  SmallVec<Value *, 4> ops;    // Call: callee first
  std::vector<uint32_t> idx;   // extract/insertvalue indices; shufflevector mask stays in ops
  SmallVec<BasicBlock *, 2> bbs; // Br successors; Phi incoming blocks (parallel to ops);
                                 // Switch: default then case destinations (ops = cond, case values)
  const AttrList *attrs = nullptr;
  Instruction(Op o, Type *t) : Value(Inst, t), op(o) {}
};

struct BasicBlock : Value {
  std::vector<Instruction *> insts;
  explicit BasicBlock(Type *labelTy) : Value(Block, labelTy) {}
};

struct Function : Value {
  Type *fty;
  std::vector<Argument *> args;
  std::vector<BasicBlock *> blocks;
  uint32_t linkage = 0, callingConv = 0, unnamedAddr = 0; // unnamedAddr: 1 global, 2 local
  const AttrList *attrs = nullptr;
  bool isDeclaration() const { return blocks.empty(); }
  Function(Type *ptrTy, Type *f) : Value(Func, ptrTy), fty(f) {}
};

// ---------------------------------------------------------------- metadata
struct Metadata {
  enum Kind : uint8_t { String, ValueMD, Node } kind;
  std::string str;
  Value *val = nullptr;          // ValueMD: a Constant or Function
  std::vector<Metadata *> ops;   // Node (nullptr allowed)
  bool distinct = false;
};
struct NamedMD {
  std::string name;
  std::vector<Metadata *> ops;
};

// ---------------------------------------------------------------- module
class Module {
public:
  std::string triple, datalayout, sourceFileName;
  std::vector<GlobalVariable *> globals;
  std::vector<Function *> functions;
  std::vector<NamedMD> namedMD;

  Module();
  Module(const Module &) = delete;
  ~Module();

  // types (uniqued, except named structs)
  Type *voidTy() const { return void_; }
  Type *halfTy() const { return half_; }
  Type *floatTy() const { return float_; }
  Type *doubleTy() const { return double_; }
  Type *labelTy() const { return label_; }
  Type *metadataTy() const { return metadata_; }
  Type *intTy(unsigned w);
  Type *ptrTy(Type *pointee, unsigned addrSpace = 0);
  Type *vecTy(Type *elem, unsigned n);
  Type *arrayTy(Type *elem, unsigned n);
  Type *fnTy(Type *ret, std::vector<Type *> params, bool vararg = false);
  Type *literalStructTy(std::vector<Type *> elems, bool packed = false);
  Type *structTy(const std::string &name);  // named, opaque until setBody
  void setBody(Type *st, std::vector<Type *> elems, bool packed = false);

  // constants (uniqued; integer/FP zero is the null value)
  Constant *constInt(Type *t, int64_t v);
  Constant *constFPBits(Type *t, uint64_t bits);
  Constant *constFloat(float f);
  Constant *constHalfBits(uint16_t bits) { return constFPBits(half_, bits); }
  Constant *constDouble(double d);
  Constant *constNull(Type *t);
  Constant *constUndef(Type *t);
  Constant *constPoison(Type *t);
  Constant *constAggregate(Type *t, std::vector<Constant *> ops);
  // Constant expressions (uniqued).
  Constant *constCast(CastOpc opc, Constant *v, Type *to);
  Constant *constGEP(Type *srcElemTy, Constant *ptr, std::vector<Constant *> idx, bool inbounds);
  Constant *constBinOp(BinOpc opc, Constant *a, Constant *b, uint8_t flags = 0);

  // globals and bodies
  Function *addFunction(const std::string &name, Type *fty);
  // Global variable. Defaults are what Metal emits for threadgroup memory: internal unnamed_addr undef.
  GlobalVariable *addGlobal(const std::string &name, Type *valueType, unsigned addrSpace, Constant *init,
                            uint32_t align, uint32_t linkage = Link_Internal, uint32_t unnamedAddr = 1,
                            bool isConstant = false);
  BasicBlock *addBlock(Function *f, const std::string &name = "");

  // attributes (uniqued)
  const AttrSet *attrSet(std::vector<Attr> attrs);
  const AttrList *attrList(std::vector<Attr> fn, std::vector<Attr> ret = {}, std::vector<std::vector<Attr>> params = {});

  // metadata (strings and uniqued nodes are uniqued by content, as in LLVM)
  Metadata *mdString(const std::string &s);
  Metadata *mdValue(Value *v);
  Metadata *mdNode(std::vector<Metadata *> ops);
  Metadata *mdDistinct(std::vector<Metadata *> ops);
  void addNamedMD(const std::string &name, std::vector<Metadata *> ops) { namedMD.push_back({name, std::move(ops)}); }

  Instruction *newInst(Op op, Type *t);

  // Reserves n consecutive tag values (never 0) for numbering values densely: a pass sets tag = base + index and tests
  // `tag - base < n` to tell its own values from stale ones.
  uint32_t allocTags(uint32_t n) {
    uint32_t base = nextTag_;
    nextTag_ += n;
    return base;
  }

private:
  uint32_t nextTag_ = 1;
  template <class T, class... A> T *make(A &&...a) {
    auto p = std::make_unique<T>(std::forward<A>(a)...);
    T *r = p.get();
    owned_.push_back(std::move(p));
    return r;
  }
  Type *newType(Type::Kind k);
  std::vector<std::unique_ptr<Type>> types_;
  std::vector<std::unique_ptr<Value>> owned_;
  std::deque<Instruction> insts_;
  std::vector<std::unique_ptr<Metadata>> mds_;
  std::vector<std::unique_ptr<AttrSet>> attrSets_;
  std::vector<std::unique_ptr<AttrList>> attrLists_;
  Type *void_, *half_, *float_, *double_, *label_, *metadata_;
  Type *intTys_[65] = {};  // by width, in front of typeIntern_
  struct ConstSlot {
    const Type *t = nullptr;
    uint64_t bits = 0;
    Constant *c = nullptr;
    bool fp = false;
  } constCache_[256];      // direct-mapped, in front of constIntern_ (constants are never removed)
  std::unordered_map<std::string, Type *> typeIntern_;
  std::unordered_map<std::string, Constant *> constIntern_;
  std::unordered_map<std::string, Metadata *> mdIntern_;
  std::unordered_map<std::string, const AttrSet *> attrSetIntern_;
  std::unordered_map<std::string, const AttrList *> attrListIntern_;
};

// ---------------------------------------------------------------- builder
class Builder {
public:
  Builder(Module &m, BasicBlock *bb) : m_(m), bb_(bb) {}
  void setInsertBlock(BasicBlock *bb) { bb_ = bb; }

  Instruction *gep(Type *srcElemTy, Value *ptr, std::vector<Value *> idx, bool inbounds, const std::string &name = "");
  Instruction *load(Type *t, Value *ptr, uint32_t align, const std::string &name = "", bool isVolatile = false);
  Instruction *store(Value *val, Value *ptr, uint32_t align, bool isVolatile = false);
  Instruction *extractValue(Value *agg, std::vector<uint32_t> idx, const std::string &name = "");
  Instruction *insertValue(Value *agg, Value *v, std::vector<uint32_t> idx, const std::string &name = "");
  Instruction *extractElement(Value *vec, Value *index, const std::string &name = "");
  Instruction *insertElement(Value *vec, Value *v, Value *index, const std::string &name = "");
  Instruction *shuffleVector(Value *a, Value *b, Constant *mask, const std::string &name = "");
  Instruction *cast(CastOpc opc, Value *v, Type *to, const std::string &name = "");
  Instruction *binop(BinOpc opc, Value *a, Value *b, uint8_t flags = 0, const std::string &name = "");
  Instruction *icmp(uint8_t pred, Value *a, Value *b, const std::string &name = "");
  Instruction *fcmp(uint8_t pred, Value *a, Value *b, uint8_t fmf = 0, const std::string &name = "");
  Instruction *select(Value *c, Value *a, Value *b, const std::string &name = "", uint8_t fmf = 0);
  Instruction *phi(Type *t, std::vector<std::pair<Value *, BasicBlock *>> incoming, const std::string &name = "",
                   uint8_t fmf = 0);
  Instruction *fneg(Value *v, uint8_t fmf = 0, const std::string &name = "");
  // Allocated in address space 0; size defaults to i32 1 (as clang emits).
  Instruction *allocaInst(Type *allocated, uint32_t align, const std::string &name = "", Value *size = nullptr);
  Instruction *switch_(Value *cond, BasicBlock *defaultDest);
  static void addCase(Instruction *sw, Constant *val, BasicBlock *dest) {
    sw->ops.push_back(val);
    sw->bbs.push_back(dest);
  }
  Instruction *call(Function *callee, const std::vector<Value *> &args, const AttrList *attrs = nullptr,
                    const std::string &name = "", uint8_t fmf = 0);
  Instruction *br(BasicBlock *dest);
  Instruction *condBr(Value *c, BasicBlock *t, BasicBlock *f);
  Instruction *retVoid();
  Instruction *ret(Value *v);
  Instruction *unreachable();

private:
  Instruction *add(Instruction *i, const std::string &name) {
    i->name = name;
    bb_->insts.push_back(i);
    return i;
  }
  Module &m_;
  BasicBlock *bb_;
};

// ---------------------------------------------------------------- writer
struct WriterOptions {
  std::string identification = "";         // IDENTIFICATION_CODE_STRING; metal-as writes ""
  std::string producer = "32023.921.6";    // irsymtab producer (metalfe-32023.921)
  bool darwinWrapper = true;               // 0x0B17C0DE header + 16-byte padding
  uint32_t cpuType = 0xFFFFFFFFu;
  bool optimizeConstants = false;          // metal-as preserves use-list order => LLVM skips this
  bool symtab = true;
  // Context-wide name tables, as registered by metalfe-32023.921's LLVMContext.
  std::vector<std::string> mdKinds = {
      "dbg", "tbaa", "prof", "fpmath", "range", "tbaa.struct", "invariant.load", "alias.scope", "noalias",
      "nontemporal", "llvm.mem.parallel_loop_access", "nonnull", "dereferenceable", "dereferenceable_or_null",
      "make.implicit", "unpredictable", "invariant.group", "align", "llvm.loop", "type", "section_prefix",
      "absolute_symbol", "associated", "callees", "irr_loop", "llvm.access.group", "callback",
      "llvm.preserve.access.index", "vcall_visibility", "noundef", "annotation"};
  std::vector<std::string> bundleTags = {"deopt", "funclet", "gc-transition", "cfguardtarget",
                                         "preallocated", "gc-live", "clang.arc.attachedcall", "ptrauth"};
  std::vector<std::string> syncScopes = {"singlethread", ""};
};

std::vector<uint8_t> writeBitcode(const Module &m, const WriterOptions &opts = {});

} // namespace airbc
