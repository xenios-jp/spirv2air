// Phi entries naming a block the module does not contain are dropped: the shader translates to the same library as
// the module without those entries. A branch to such a block is an error, not a block without a terminator.
#include <spirv2air.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr uint32_t kOpLabel = 248, kOpBranch = 249, kOpPhi = 245;

std::unordered_set<uint32_t> labels(const std::vector<uint32_t>& words) {
  std::unordered_set<uint32_t> out;
  for (size_t pos = 5; pos < words.size(); pos += words[pos] >> 16)
    if ((words[pos] & 0xFFFF) == kOpLabel) out.insert(words[pos + 1]);
  return out;
}

// The module with every phi entry whose parent is not a label removed; `missing` gets the first such parent.
std::vector<uint32_t> withoutDangling(const std::vector<uint32_t>& words, uint32_t& missing) {
  const std::unordered_set<uint32_t> defined = labels(words);
  std::vector<uint32_t> out(words.begin(), words.begin() + 5);
  for (size_t pos = 5; pos < words.size(); pos += words[pos] >> 16) {
    uint32_t wc = words[pos] >> 16, op = words[pos] & 0xFFFF;
    if (op != kOpPhi) {
      out.insert(out.end(), words.begin() + long(pos), words.begin() + long(pos + wc));
      continue;
    }
    size_t at = out.size();
    out.insert(out.end(), words.begin() + long(pos), words.begin() + long(pos + 3));
    for (uint32_t w = 3; w + 1 < wc; w += 2) {
      uint32_t parent = words[pos + w + 1];
      if (!defined.count(parent)) {
        if (!missing) missing = parent;
        continue;
      }
      out.push_back(words[pos + w]);
      out.push_back(parent);
    }
    out[at] = uint32_t(out.size() - at) << 16 | kOpPhi;
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  if (words.size() < 5) return 2;
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);

  auto dangling = spirv2air::translate(words.data(), words.size());
  if (!dangling.ok()) return std::fprintf(stderr, "dangling phi entries: %s\n", dangling.error.c_str()), 1;
  uint32_t missing = 0;
  std::vector<uint32_t> clean = withoutDangling(words, missing);
  if (!missing || clean.size() >= words.size()) return std::fprintf(stderr, "the fixture has no dangling entry\n"), 2;
  auto reference = spirv2air::translate(clean.data(), clean.size());
  if (!reference.ok()) return std::fprintf(stderr, "without the entries: %s\n", reference.error.c_str()), 1;
  if (dangling.metallib != reference.metallib) {
    std::fprintf(stderr, "dropping the dangling entries changed the library\n");
    return 1;
  }

  // Point the first OpBranch at the missing block.
  for (size_t pos = 5; pos < words.size(); pos += words[pos] >> 16)
    if ((words[pos] & 0xFFFF) == kOpBranch) {
      words[pos + 1] = missing;
      break;
    }
  auto branch = spirv2air::translate(words.data(), words.size());
  const std::string want = "label %" + std::to_string(missing) + " is not defined in the function";
  if (branch.ok() || branch.error.find(want) == std::string::npos) {
    std::fprintf(stderr, "a branch to a missing block was not rejected (%s)\n", branch.error.c_str());
    return 1;
  }
  std::puts("PASS: dangling phi entries dropped, branch to a missing block rejected");
  return 0;
}
