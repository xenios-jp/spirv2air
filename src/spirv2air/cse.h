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
// With forwardPrivate, loads from allocas whose address never escapes also take the value last stored or loaded
// through the same pointer when nothing can have written the alloca since, and stores that are overwritten in the same
// block before any read, or that store what the alloca already holds, are removed (see cse.cpp).
void cseFunction(airbc::Module& m, airbc::Function* f, bool forwardPrivate = false);

}  // namespace spirv2air
