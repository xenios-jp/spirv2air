// Branch conditions the constants decide, found without touching the rest of the function.
#pragma once

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// One forward walk over the blocks reachable from the entry, in reverse post-order, that follows only the edges a
// branch can take: a `br` or `switch` whose condition is a value computed from constants alone (integer arithmetic,
// compares, selects, casts, bit-field extracts, lanes of constant vectors, phis whose reachable entries agree) gets that
// constant as its condition, so that cleanupFunction folds it and drops the arms it rules out. Nothing else changes:
// no instruction is removed or replaced, and a phi is read only through the edges already known to be taken, with the
// entries that arrive over a back edge counted as unknown. Returns whether any condition was replaced.
//
// The point is cost, not reach: the rounds of simplifyInstructions and cleanupFunction find the same branches, but they
// do it by folding and sweeping every instruction of the arms that are about to disappear (most of a baked shader).
// This finds them first, from the conditions alone, so those rounds start on the function that is left.
bool foldKnownBranches(airbc::Module& m, airbc::Function* f);

}  // namespace spirv2air
