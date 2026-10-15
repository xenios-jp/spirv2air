// CFG and dead-code cleanup of the AIR spirv2air emits.
#pragma once
#include <string>

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// What one run of cleanupFunction removed.
struct CleanupStats {
  unsigned unreachableBlocks = 0;  // blocks no path from the entry reaches
  unsigned foldedBranches = 0;     // constant or same-target br / switch turned into an unconditional br
  unsigned mergedBlocks = 0;       // blocks merged into their single predecessor
  unsigned trivialPhis = 0;        // phis whose incoming values are all one value
  unsigned forwardedExtracts = 0;  // extractelements replaced by the element they read
  unsigned deadPhis = 0;           // phis nothing live uses
  unsigned deadInsts = 0;          // other side-effect-free instructions nothing live uses
};

// Removes unreachable blocks, folds constant and redundant branches, merges straight-line blocks, drops trivial and
// unused phis, and deletes side-effect-free instructions whose result is unused, until nothing changes. Computes
// exactly what the input computes: no constant folding of arithmetic, no reordering, flags untouched. The entry stays
// the first block and the others keep their relative order.
CleanupStats cleanupFunction(airbc::Module& m, airbc::Function* f);

// Structural check of a function's IR: every block ends in exactly one terminator, phis only at block starts with
// exactly one entry per predecessor edge, branch targets and instruction operands defined in blocks of the function
// (and, within a block, before their use). Returns an empty string, or a description of the first problem.
std::string verifyFunction(const airbc::Function* f);

}  // namespace spirv2air
