// The opt-in attachment ABI accepts only ordered, current-sample float4 reads.
#include <spirv2air.h>
#include <cstdio>
#include <fstream>
#include <vector>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  if (words.size() < 5) return 2;
  file.seekg(0); file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  spirv2air::Options options;
  options.graphics.framebufferFetchMask = 1;
  auto run = [&](const std::vector<uint32_t>& w) {
    return spirv2air::translate(w.data(), w.size(), options);
  };
  auto good = run(words);
  if (!good.ok()) { puts(good.error.c_str()); return 1; }
  if (good.metallib != run(words).metallib) return 1;
  if (spirv2air::translate(words.data(), words.size()).ok()) return 1;
  uint32_t one = 0;
  for (size_t p = 5; p < words.size(); p += words[p] >> 16)
    if ((words[p] & 65535) == 43 && words[p + 3] == 1) one = words[p + 2];
  if (!one) return 2;
  // A fixed sample, a nonzero coordinate, a missing output, an arrayed image,
  // and an unmapped attachment must each produce a diagnostic, not a library.
  for (int test = 0; test < 5; ++test) {
    auto bad = words;
    for (size_t p = 5; p < bad.size(); p += bad[p] >> 16) {
      unsigned op = bad[p] & 65535;
      if (test == 0 && op == 98) bad[p + 6] = one; // OpImageRead sample
      if (test == 1 && op == 98) bad[p + 4] = one; // OpImageRead coordinate
      if (test == 2 && op == 71 && bad[p + 2] == 30) bad[p + 3] = 1; // Location
      if (test == 3 && op == 25) bad[p + 5] = 1; // OpTypeImage Arrayed
      if (test == 4 && op == 71 && bad[p + 2] == 43) bad[p + 3] = 1; // AttachmentIndex
    }
    auto r = run(bad);
    if (r.ok() || r.error.empty()) { printf("FAIL rejection %d\n", test); return 1; }
  }
  puts("PASS: deterministic opt-in feedback and five unsupported forms rejected");
}
