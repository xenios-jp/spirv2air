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
// forwarded. Calls of the integer bit intrinsics (extract_bits, insert_bits, integer min/max) on constants are
// evaluated where Metal defines their result. No floating-point arithmetic is evaluated or rewritten: only fneg of a
// constant (a sign-bit flip), the bit reinterpretation of constants, and conversions and roundings whose result is an
// exact integer under any rounding mode and fast-math setting (an integer constant that f32 holds exactly; rint, trunc,
// floor, ceil of a constant that is already an integer; Apple's float -> int conversion of an in-range integer). The replaced instructions stay (dead) for cleanupFunction to sweep. Returns
// whether anything was replaced.
bool simplifyInstructions(airbc::Module& m, airbc::Function* f);

}  // namespace spirv2air
