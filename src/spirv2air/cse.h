// Common-subexpression elimination of the AIR spirv2air emits.
#pragma once

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// Walks the dominator tree once and replaces every pure instruction (arithmetic, casts, compares, selects, GEPs, vector
// and aggregate element operations, readnone calls) that is identical to a dominating one, operands included, by that
// one. Loads from read-only memory (address space 2, and address space 1 when the function never writes device memory)
// are treated the same way. Nothing is evaluated or reordered, flags must match exactly, and a value is only ever
// replaced by one that dominates it, so every remaining value is computed as before. Replaced instructions are removed.
void cseFunction(airbc::Module& m, airbc::Function* f);

}  // namespace spirv2air
