// Private memory forwarding in cseFunction: for every case of tests/forward_cases.h, the private loads and stores left
// after the pass are exactly the ones the case says must stay (fewer means a forward or a store removal that is wrong
// for some row; more means a missed one), and without forwarding none is removed. forward_gpu checks the values.
// usage: forward_cpu
#include <cstdio>

#include "forward_cases.h"

int main() {
  int failures = 0;
  for (const forwardcases::Case& c : forwardcases::cases()) {
    airkit::Kernel off("k"), on("k");
    forwardcases::build(off, c);
    forwardcases::build(on, c);
    const auto before = forwardcases::privateAccesses(off.function());
    spirv2air::cseFunction(off.m, off.function(), false);
    spirv2air::cseFunction(on.m, on.function(), true);
    const auto plain = forwardcases::privateAccesses(off.function());
    const auto [loads, stores] = forwardcases::privateAccesses(on.function());
    const bool bad = loads != c.loads || stores != c.stores || plain != before;
    std::printf("%s %s: private loads %u -> %u (want %u), stores %u -> %u (want %u)%s\n", bad ? "FAIL" : "ok  ",
                c.name.c_str(), before.first, loads, c.loads, before.second, stores, c.stores,
                plain != before ? ", changed without forwarding" : "");
    failures += bad;
  }
  std::printf(failures ? "%d failing\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
