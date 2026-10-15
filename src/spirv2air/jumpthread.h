// Jump threading of the Xenos control-flow dispatch loop.
#pragma once
#include <string>

namespace airbc {
class Module;
struct Function;
}  // namespace airbc

namespace spirv2air {

// What one run of jumpThread changed.
struct JumpThreadStats {
  unsigned dispatchers = 0;       // blocks of phis ending in a switch on one of them that were threaded
  unsigned dissolvedLatches = 0;  // phi-only forwarding blocks into a dispatcher, bypassed by their predecessors
  unsigned threadedEdges = 0;     // edges into a dispatcher redirected to the block its switch picks
  unsigned splitSelects = 0;      // ... of which the dispatch value was a select of two constants (br on its condition)
  unsigned phis = 0;              // phis inserted to carry the dispatcher's phi values into the targets
};

// A block made only of phis and a `switch` on one of them (the main loop header of xenia's Xenos translation: `pc`
// phi, switch on it) is a jump table whose selector is known on every edge that carries a constant. Such an edge is
// redirected straight to the block the switch would pick (a select of two constants becomes a conditional branch on
// its condition), after the dispatcher's phi-only latch is bypassed so its predecessors become the dispatcher's. Values
// of the dispatcher's phis along a redirected edge are the phi entry of that edge; SSA form is rebuilt by phis in the
// targets and at the iterated dominance frontier (trivial and unused ones are left to cleanupFunction). Only edges
// change: every instruction runs on exactly the paths and with exactly the values it did, and nothing is duplicated.
// Returns whether anything changed; run cleanupFunction afterwards.
bool jumpThread(airbc::Module& m, airbc::Function* f, JumpThreadStats* stats = nullptr);

// Checks that every use is dominated by its definition (phi operands: at the end of the incoming block). Returns an
// empty string, or a description of the first violation. For debug builds; verifyFunction does not look at dominance.
std::string verifyDominance(const airbc::Function* f);

}  // namespace spirv2air
