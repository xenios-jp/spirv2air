// Control-flow simplification of the AIR spirv2air emits.
#pragma once

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// What one run of simplifyCfg changed.
struct SimplifyCfgStats {
  unsigned loweredSwitches = 0;  // switches turned into a chain of tests, one per case
  unsigned forwardedBlocks = 0;  // empty blocks (only a `br`) bypassed by retargeting their predecessors
  unsigned convertedArms = 0;    // single-predecessor arms of an if / if-else hoisted and joined by selects
};

// Bypasses empty forwarding blocks, and turns small if / if-else shapes whose arms hold only side-effect-free,
// non-trapping instructions into straight-line code: the arms' instructions move into the branching block (they
// run on both paths, nothing reads their results on the other) and each phi of the join becomes a select on the
// branch condition. Only control-flow edges change and phis become selects of the same two values, so every
// result is computed exactly as before. Returns whether anything changed; run cleanupFunction afterwards to merge
// the blocks that are left in a line, resolve single-entry phis and sweep.
bool simplifyCfg(airbc::Module& m, airbc::Function* f, SimplifyCfgStats* stats = nullptr);

}  // namespace spirv2air
