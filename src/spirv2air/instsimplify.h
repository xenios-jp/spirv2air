// Exact instruction simplification of the AIR spirv2air emits.
#pragma once

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// Replaces instructions by a value that provably equals their result bit for bit: integer and bool operations,
// compares and casts of constants are evaluated; identities (x+0, x&x, x&mask when x already fits mask, shift by 0,
// compare of a value with itself, select with a constant or equal arms, bitcast to the same type, double fneg, ...) are
// forwarded. No floating-point arithmetic is evaluated or rewritten: only fneg of a constant (a sign-bit flip) and the
// bit reinterpretation of constants. The replaced instructions stay (dead) for cleanupFunction to sweep. Returns
// whether anything was replaced.
bool simplifyInstructions(airbc::Module& m, airbc::Function* f);

}  // namespace spirv2air
