// simplifyInstructions' constant folding of AIR intrinsic calls and int -> float casts: every tuple of
// tests/fold_cases.h folds exactly when the operation's result is defined bit for bit, to the host reference value.
// fold_gpu checks the same folds against Apple's compiler and GPU.
// usage: fold_cpu
#include <cstdio>

#include "fold_cases.h"

int main() {
  int failures = 0;
  size_t tuples = 0, folds = 0;
  for (const foldcases::Case& c : foldcases::cases()) {
    airkit::Kernel k("k");
    auto stores = foldcases::buildConstant(k, c);
    int bad = foldcases::checkFolds(k, c, stores);
    for (const auto& t : c.tuples) folds += t.fold;
    tuples += c.tuples.size();
    std::printf("%s %s: %zu tuples\n", bad ? "FAIL" : "ok  ", c.name.c_str(), c.tuples.size());
    failures += bad != 0;
  }
  std::printf("%zu tuples, %zu fold\n", tuples, folds);
  std::printf(failures ? "%d failing\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
