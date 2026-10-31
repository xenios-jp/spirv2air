// Pixel interlock translates only with a raster-ordered buffer configured, and
// then deterministically, with that buffer in raster order group 0.
#include <algorithm>
#include <cstdio>
#include <cstring>
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
  auto rejected = spirv2air::translate(words.data(), words.size());
  if (rejected.ok() || rejected.error.find("requires graphics.rasterOrderedBuffer") == std::string::npos) {
    std::fprintf(stderr, "pixel interlock without a raster-ordered buffer was not rejected\n");
    return 1;
  }
  spirv2air::Options options;
  options.graphics.rasterOrderedBuffer = 8;
  auto good = spirv2air::translate(words.data(), words.size(), options);
  if (!good.ok()) return std::fprintf(stderr, "%s\n", good.error.c_str()), 1;
  if (good.metallib != spirv2air::translate(words.data(), words.size(), options).metallib) return 1;
  const char tag[] = "air.raster_order_group";
  if (std::search(good.metallib.begin(), good.metallib.end(), tag, tag + std::strlen(tag)) == good.metallib.end()) {
    std::fprintf(stderr, "no raster order group in the translated library\n");
    return 1;
  }
  std::puts("PASS: pixel interlock on the raster-ordered buffer");
  return 0;
}
