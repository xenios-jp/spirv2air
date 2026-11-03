// Pull-model interpolation translates deterministically. InterpolateAt* on a Flat input reads the flat value and
// needs no interpolant; a Sample-decorated operand and a builtin operand are rejected. The raster-coverage centroid
// fixture's subgroup vote becomes air.simd_any, and a vote at another scope is rejected.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <spirv2air.h>
#include <string>
#include <vector>

namespace {

bool contains(const std::vector<uint8_t>& bytes, const std::string& s) {
  return std::search(bytes.begin(), bytes.end(), s.begin(), s.end()) != bytes.end();
}

std::vector<uint32_t> read(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  return words;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::vector<uint32_t> words = read(argv[1]), centroid = read(argv[2]);
  if (words.size() < 5 || centroid.size() < 5) return 2;
  auto translate = [](const std::vector<uint32_t>& data) { return spirv2air::translate(data.data(), data.size()); };
  auto good = translate(words);
  if (!good.ok()) return std::fprintf(stderr, "%s\n", good.error.c_str()), 1;
  if (good.metallib != translate(words).metallib) return std::fprintf(stderr, "not deterministic\n"), 1;
  for (const char* name : {"air.interpolate_offset_perspective.v4f32", "air.interpolate_sample_no_perspective.v4f32",
                           "air.interpolate_centroid_no_perspective.v4f32",
                           "air.interpolate_centroid_perspective.v4f32", "air.interpolate_center_no_perspective.v4f32"})
    if (!contains(good.metallib, name)) return std::fprintf(stderr, "no %s\n", name), 1;

  // The fixture's inputs: a (Centroid) and b (NoPerspective); declarations start at the first OpTypeVoid.
  uint32_t a = 0, b = 0;
  size_t declarations = 0;
  for (size_t p = 5; p < words.size(); p += words[p] >> 16) {
    if ((words[p] & 65535) == 71 && (words[p] >> 16) == 3 && words[p + 2] == 16) a = words[p + 1];
    if ((words[p] & 65535) == 71 && (words[p] >> 16) == 3 && words[p + 2] == 13) b = words[p + 1];
    if (!declarations && (words[p] & 65535) == 19) declarations = p;
  }
  if (!a || !b || !declarations) return std::fprintf(stderr, "fixture inputs not found\n"), 2;
  auto decorate = [&](uint32_t id, std::vector<uint32_t> operands) {
    auto w = words;
    operands.insert(operands.begin(), {uint32_t((2 + operands.size()) << 16 | 71), id});
    w.insert(w.begin() + declarations, operands.begin(), operands.end());
    return translate(w);
  };
  // Flat b: InterpolateAtSample and InterpolateAtCentroid of b read its value; only a stays an interpolant.
  auto flat = decorate(b, {14});
  if (!flat.ok()) return std::fprintf(stderr, "flat: %s\n", flat.error.c_str()), 1;
  if (contains(flat.metallib, "no_perspective") || !contains(flat.metallib, "air.interpolate_offset_perspective"))
    return std::fprintf(stderr, "flat input is still an interpolant\n"), 1;
  auto sample = decorate(a, {17});
  if (sample.ok() || sample.error.find("Sample interpolation") == std::string::npos)
    return std::fprintf(stderr, "Sample-decorated operand was not rejected\n"), 1;
  auto builtin = decorate(b, {11, 15});  // BuiltIn FragCoord
  if (builtin.ok() || builtin.error.find("not a fragment input") == std::string::npos)
    return std::fprintf(stderr, "builtin operand was not rejected: %s\n", builtin.error.c_str()), 1;

  auto vote = translate(centroid);
  if (!vote.ok() || vote.metallib != translate(centroid).metallib || !contains(vote.metallib, "air.simd_any") ||
      !contains(vote.metallib, "air.interpolate_sample_perspective.v4f32"))
    return std::fprintf(stderr, "raster-coverage centroid: %s\n", vote.error.c_str()), 1;
  // The vote's scope (OpGroupNonUniformAny = 335) set to the constant 4, Invocation.
  uint32_t four = 0;
  for (size_t p = 5; p < centroid.size(); p += centroid[p] >> 16)
    if ((centroid[p] & 65535) == 43 && (centroid[p] >> 16) == 4 && centroid[p + 3] == 4) four = centroid[p + 2];
  bool voted = false;
  for (size_t p = 5; p < centroid.size(); p += centroid[p] >> 16)
    if ((centroid[p] & 65535) == 335) centroid[p + 3] = four, voted = true;
  if (!four || !voted) return std::fprintf(stderr, "no vote or constant 4 in the centroid fixture\n"), 2;
  auto scope = translate(centroid);
  if (scope.ok() || scope.error.find("group operation scope") == std::string::npos)
    return std::fprintf(stderr, "a vote at another scope was not rejected\n"), 1;
  std::puts("PASS: pull-model interpolation, flat operands, operand rejection and the subgroup vote");
  return 0;
}
