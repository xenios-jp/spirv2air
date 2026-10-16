// Packing policy is explicit, deterministic, and part of the output identity.
#include <spirv2air.h>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
  if (!input || input.tellg() < 20 || input.tellg() % 4) return 2;
  std::vector<uint32_t> words(size_t(input.tellg()) / 4);
  input.seekg(0); input.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  spirv2air::Options options; options.bindings = {{0,0,0,0}, {0,1,1,0}};
  auto run = [&] { return spirv2air::translate(words.data(), words.size(), options); };
  auto original = run();
  options.halfPackingRounding = spirv2air::HalfPackingRounding::NearestEven;
  auto nearest = run();
  if (!original.ok() || !nearest.ok() || original.metallib != nearest.metallib) return 1;
  options.halfPackingRounding = spirv2air::HalfPackingRounding::TowardZero;
  auto towardZero = run(), repeated = run();
  if (!towardZero.ok() || !repeated.ok() || towardZero.metallib != repeated.metallib ||
      towardZero.metallib == nearest.metallib) return 1;
  options.halfPackingRounding = static_cast<spirv2air::HalfPackingRounding>(255);
  auto invalid = run();
  if (invalid.ok() || invalid.error.find("half packing rounding") == std::string::npos) return 1;
  std::puts("half_packing_cpu: defaults, policy identity, determinism and invalid policy checked");
  return 0;
}
