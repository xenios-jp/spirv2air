// Scalarization of the elementwise vector code spirv2air emits.
#pragma once

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// Splits every elementwise vector instruction (binary operations, fneg, compares, selects, casts that keep the lane
// count, phis, and insertelement / shufflevector with constant indices) into one scalar instruction per lane, flags
// included, so each lane computes exactly what it did as part of the vector. Lanes of values that stay vectors (loads,
// calls, arguments) are read with extractelement; vectors that stores, calls and the like still use are rebuilt with an
// insertelement chain. Leaves the leftover vectors to cleanupFunction, which forwards extracts and deletes them.
void scalarizeFunction(airbc::Module& m, airbc::Function* f);

}  // namespace spirv2air
