// KeepBranch (NonSemantic.spirv2air) keeps a branch that would otherwise become a select. simplifyCfg leaves a
// conditional br marked kBrKeep as a branch and a phi, and if-converts the same shape without the mark. Translating
// the fixture (a one-instruction pure arm, KeepBranch in its header, selection control DontFlatten) gives a different
// library from the same module without KeepBranch; without KeepBranch the selection control changes nothing
// (DontFlatten, None and Flatten all give the same library, the select).
// usage: keepbranch_cpu keepbranch.frag.spv
#include <spirv2air.h>

#include <cstdio>
#include <fstream>
#include <vector>

#include "airkit/airkit.h"
#include "spirv2air/simplifycfg.h"

namespace {

using airbc::BasicBlock;
using airbc::Builder;
using airbc::Op;
using airbc::Value;

constexpr uint32_t kOpNop = 0, kOpExtInst = 12, kOpSelectionMerge = 247;

struct Shape {
  unsigned blocks = 0, phis = 0, selects = 0, converted = 0;
};

// out[g] = in[g] < 0 ? fabs(in[g]) : in[g], as a triangle with a one-instruction arm, then simplifyCfg.
Shape run(bool keep) {
  airkit::Kernel k("k");
  unsigned in = k.buffer(k.f32, 0, airkit::Access::Read, "in");
  unsigned out = k.buffer(k.f32, 1, airkit::Access::ReadWrite, "out");
  unsigned gid = k.builtin(airkit::Builtin::ThreadPositionInGrid);
  k.begin();
  Builder& b = k.builder();
  Value* g = b.cast(airbc::ZExt, k.arg(gid), k.i64);
  Value* x = b.load(k.f32, b.gep(k.f32, k.arg(in), {g}, true), 4);
  BasicBlock* entry = k.function()->blocks.back();
  BasicBlock *then = k.block("then"), *merge = k.block("merge");
  airbc::Instruction* br = b.condBr(b.fcmp(4 /* olt */, x, k.m.constFloat(0.0f)), then, merge);
  if (keep) br->flags |= spirv2air::kBrKeep;
  b.setInsertBlock(then);
  Value* a = k.math("fabs", x, false);
  b.br(merge);
  b.setInsertBlock(merge);
  Value* r = b.phi(k.f32, {{a, then}, {x, entry}});
  b.store(r, b.gep(k.f32, k.arg(out), {g}, true), 4);
  b.retVoid();

  spirv2air::SimplifyCfgStats stats;
  spirv2air::simplifyCfg(k.m, k.function(), &stats);
  Shape s;
  s.converted = stats.convertedArms;
  for (const BasicBlock* bb : k.function()->blocks) {
    ++s.blocks;
    for (const airbc::Instruction* i : bb->insts) {
      s.phis += i->op == Op::Phi;
      s.selects += i->op == Op::Select;
    }
  }
  return s;
}

// The fixture with the control of its OpSelectionMerge replaced, and without its KeepBranch (made OpNops).
std::vector<uint32_t> variant(std::vector<uint32_t> words, uint32_t control, bool keep, bool& found) {
  for (size_t pos = 5; pos < words.size(); pos += words[pos] >> 16) {
    uint32_t wc = words[pos] >> 16, op = words[pos] & 0xFFFF;
    if (op == kOpSelectionMerge) words[pos + 2] = control;
    // OpExtInst %void %ns 1: 5 words, the only extended instruction without operands.
    if (op == kOpExtInst && wc == 5 && words[pos + 4] == spirv2air::kSpirv2airKeepBranch) {
      found = true;
      if (!keep)
        for (uint32_t w = 0; w < wc; ++w) words[pos + w] = 1u << 16 | kOpNop;
    }
  }
  return words;
}

}  // namespace

int main(int argc, char** argv) {
  int failures = 0;
  const Shape kept = run(true), converted = run(false);
  std::printf("marked:   %u blocks, %u phis, %u selects, %u arms converted\n", kept.blocks, kept.phis, kept.selects,
              kept.converted);
  std::printf("unmarked: %u blocks, %u phis, %u selects, %u arms converted\n", converted.blocks, converted.phis,
              converted.selects, converted.converted);
  if (kept.converted || kept.selects || kept.phis != 1 || kept.blocks < 3) {
    std::printf("FAIL the marked branch was changed\n");
    ++failures;
  }
  if (converted.converted != 1 || converted.selects != 1) {
    std::printf("FAIL the unmarked branch was not if-converted\n");
    ++failures;
  }

  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  if (words.size() < 5) return 2;
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  bool found = false;
  std::vector<uint32_t> keep = variant(words, 2, true, found);
  std::vector<uint32_t> dontFlatten = variant(words, 2, false, found), none = variant(words, 0, false, found),
                        flatten = variant(words, 1, false, found);
  if (!found) return std::fprintf(stderr, "the fixture has no KeepBranch\n"), 2;
  auto k = spirv2air::translate(keep.data(), keep.size());
  auto d = spirv2air::translate(dontFlatten.data(), dontFlatten.size());
  auto n = spirv2air::translate(none.data(), none.size());
  auto f = spirv2air::translate(flatten.data(), flatten.size());
  if (!k.ok() || !d.ok() || !n.ok() || !f.ok()) {
    std::fprintf(stderr, "translation failed: %s|%s|%s|%s\n", k.error.c_str(), d.error.c_str(), n.error.c_str(),
                 f.error.c_str());
    return 1;
  }
  std::printf("library bytes: KeepBranch %zu; without it: DontFlatten %zu, None %zu, Flatten %zu\n", k.metallib.size(),
              d.metallib.size(), n.metallib.size(), f.metallib.size());
  if (k.metallib == d.metallib) {
    std::printf("FAIL KeepBranch did not change the library\n");
    ++failures;
  }
  if (d.metallib != n.metallib || n.metallib != f.metallib) {
    std::printf("FAIL without KeepBranch the selection control changed the library\n");
    ++failures;
  }
  std::printf(failures ? "%d failing\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
