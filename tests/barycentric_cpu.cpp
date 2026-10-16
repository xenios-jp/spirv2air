// Center barycentrics translate deterministically; unsupported sampling and
// malformed builtin types must fail before generating a Metal library.
#include <cstdio>
#include <fstream>
#include <spirv2air.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  if (words.size() < 5) return 2;
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  auto translate = [](const std::vector<uint32_t>& data) { return spirv2air::translate(data.data(), data.size()); };
  auto good = translate(words);
  if (!good.ok() || good.metallib != translate(words).metallib) return 1;
  auto perspective = words;
  for (size_t p = 5; p < perspective.size(); p += perspective[p] >> 16) {
    if ((perspective[p] & 65535) == 71 && (perspective[p] >> 16) == 4 && perspective[p + 2] == 11 &&
        perspective[p + 3] == 5287)
      perspective[p + 3] = 5286;
  }
  auto unsupported_perspective = translate(perspective);
  if (unsupported_perspective.ok() ||
      unsupported_perspective.error.find("perspective barycentric input is not supported") == std::string::npos)
    return 1;
  uint32_t input = 0;
  size_t declarations = 0;
  for (size_t p = 5; p < words.size(); p += words[p] >> 16) {
    if ((words[p] & 65535) == 71 && (words[p] >> 16) == 4 && words[p + 2] == 11 && words[p + 3] == 5287)
      input = words[p + 1];
    if (!declarations && (words[p] & 65535) == 19) declarations = p;
  }
  if (!input || !declarations) return 2;
  for (uint32_t decoration : {14u, 16u, 17u}) {
    auto unsupported = words;
    unsupported.insert(unsupported.begin() + declarations, {uint32_t((3 << 16) | 71), input, decoration});
    auto result = translate(unsupported);
    if (result.ok() || result.error.find("center sampling") == std::string::npos) {
      std::fprintf(stderr, "unsupported barycentric sampling was not rejected\n");
      return 1;
    }
  }
  auto malformed = words;
  for (size_t p = 5; p < malformed.size(); p += malformed[p] >> 16) {
    if ((malformed[p] & 65535) == 23 && malformed[p + 3] == 3) malformed[p + 3] = 2;
  }
  auto result = translate(malformed);
  if (result.ok() || result.error.find("must be float3") == std::string::npos) return 1;
  std::puts("PASS: center barycentrics and unsupported-input rejection");
  return 0;
}
