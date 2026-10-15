// SSA promotion of allocas (LLVM's mem2reg) for the AIR spirv2air emits.
#pragma once

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// Promotes every entry-block alloca whose only uses are whole-value, non-volatile loads and stores of its allocated
// type to SSA values, inserting phis at the iterated dominance frontier of its stores. A load before any store reads
// undef, as the memory would. Other allocas and every other instruction are left unchanged.
void promoteAllocas(airbc::Module& m, airbc::Function* f);

}  // namespace spirv2air
