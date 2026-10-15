// Translate one SPIR-V module. All file I/O stays in this executable; the library returns bytes.
#include <spirv2air.h>
#include <charconv>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static uint32_t number(const std::string& text) {
  uint32_t value = 0;
  auto r = std::from_chars(text.data(), text.data() + text.size(), value);
  if (r.ec != std::errc{} || r.ptr != text.data() + text.size()) throw std::runtime_error("expected an unsigned decimal integer");
  return value;
}
static void binding(spirv2air::Options& opts, const std::string& text, bool texture) {
  auto a = text.find(':'), b = a == text.npos ? text.npos : text.find(':', a + 1);
  if (a == text.npos || b == text.npos) throw std::runtime_error("binding must be SET:BINDING:INDEX");
  auto set = number(text.substr(0, a)), slot = number(text.substr(a + 1, b - a - 1)), index = number(text.substr(b + 1));
  for (auto& entry : opts.bindings) if (entry.set == set && entry.binding == slot) {
    if (texture) entry.texture = index; else entry.buffer = index;
    return;
  }
  spirv2air::ResourceBinding entry;
  entry.set = set; entry.binding = slot;
  if (texture) entry.texture = index; else entry.buffer = index;
  opts.bindings.push_back(entry);
}
static spirv2air::StorageBufferSlot storageSlot(const std::string& text) {
  auto a = text.find(':'), b = a == text.npos ? text.npos : text.find(':', a + 1);
  if (a == text.npos || b == text.npos) throw std::runtime_error("storage buffer must be SET:BINDING:OFFSET");
  return {number(text.substr(0, a)), number(text.substr(a + 1, b - a - 1)), number(text.substr(b + 1))};
}
// Batch rows: <source-key> <stage> <variant> <SPIR-V path>.
// The source-key basename (without its suffix), stage and variant determine the output name.
static int manifest(const char* input, const char* output, const spirv2air::Options& opts = {}) {
  std::ifstream rows(input);
  if (!rows) throw std::runtime_error("cannot open manifest");
  std::filesystem::create_directories(output);
  std::map<std::string, size_t> failures;
  size_t total = 0, ok = 0;
  std::string line;
  while (std::getline(rows, line)) {
    std::istringstream fields(line);
    std::string source, stage, variant, path;
    if (!(fields >> source >> stage >> variant >> path)) continue;
    ++total;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in || in.tellg() < 20 || in.tellg() % 4) { ++failures["unreadable SPIR-V"]; continue; }
    std::vector<uint32_t> words(size_t(in.tellg()) / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(words.data()), words.size() * 4)) {
      ++failures["unreadable SPIR-V"]; continue;
    }
    auto result = spirv2air::translate(words.data(), words.size(), opts);
    if (!result.ok()) { ++failures[result.error]; continue; }
    auto key = std::filesystem::path(source).filename().string();
    key = key.substr(0, key.find('.'));
    if (key.empty() || key == "." || key == ".." || stage.find_first_of("/\\") != stage.npos ||
        variant.find_first_of("/\\") != variant.npos) throw std::runtime_error("invalid output name in manifest");
    auto file = std::filesystem::path(output) / (key + "." + stage + "." + variant + ".metallib");
    std::ofstream out(file, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(result.metallib.data()), result.metallib.size())) {
      ++failures["cannot write output"]; continue;
    }
    out.close();
    if (!out) { ++failures["cannot write output"]; continue; }
    ++ok;
  }
  if (rows.bad()) throw std::runtime_error("cannot finish reading manifest");
  std::printf("%zu/%zu translated\n", ok, total);
  for (auto& [reason, count] : failures) std::printf("%6zu  %s\n", count, reason.c_str());
  return ok == total ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") { std::puts(spirv2air::kVersion); return 0; }
  if (argc == 2 && std::string(argv[1]) == "--help") {
    std::puts(
        "usage: spirv2air_cli INPUT.spv OUTPUT.metallib [--buffer SET:BINDING:INDEX] [--texture SET:BINDING:INDEX] "
        "[--push-constant INDEX] [--no-jump-thread] [--function-constant-set SET] [--storage-buffer SET:BINDING:OFFSET] [--platform macos|ios|ios-simulator] [--min-os MAJOR[.MINOR]]");
    std::puts("       spirv2air_cli --manifest INPUT.txt OUTPUT_DIR [--function-constant-set SET] [--storage-buffer SET:BINDING:OFFSET]\n       spirv2air_cli INPUT.txt OUTPUT_DIR (legacy batch form)");
    return 0;
  }
  if (argc < 3) { std::fprintf(stderr, "usage: spirv2air_cli INPUT.spv OUTPUT.metallib [options]; see --help\n"); return 2; }
  try {
    if (std::string(argv[1]) == "--manifest") {
      spirv2air::Options opts;
      for (int i = 4; i < argc; i += 2) {
        std::string flag = argv[i];
        if (i + 1 >= argc) throw std::runtime_error("missing argument: " + flag);
        if (flag == "--function-constant-set") opts.graphics.functionConstantSet = int32_t(number(argv[i + 1]));
        else if (flag == "--storage-buffer") opts.graphics.storageBuffers.push_back(storageSlot(argv[i + 1]));
        else throw std::runtime_error("--manifest accepts [--function-constant-set N] [--storage-buffer SET:BINDING:OFFSET]");
      }
      return manifest(argv[2], argv[3], opts);
    }
    // Preserve the original two-argument batch interface for text manifests.
    if (argc == 3 && std::filesystem::path(argv[1]).extension() == ".txt") return manifest(argv[1], argv[2]);
    spirv2air::Options opts;
    for (int i = 3; i < argc; ++i) {
      std::string flag = argv[i];
      if (flag == "--no-jump-thread") opts.jumpThread = false;
 else if (flag == "--storage-buffer" && i + 1 < argc) opts.graphics.storageBuffers.push_back(storageSlot(argv[++i]));
      else if (flag == "--function-constant-set" && i + 1 < argc) opts.graphics.functionConstantSet = int32_t(number(argv[++i]));
      else if (flag == "--platform" && i + 1 < argc) {
        std::string value = argv[++i];
        if (value == "macos") opts.platform = spirv2air::Platform::MacOS;
        else if (value == "ios") opts.platform = spirv2air::Platform::IOS;
        else if (value == "ios-simulator") opts.platform = spirv2air::Platform::IOSSimulator;
        else throw std::runtime_error("unknown platform: " + value);
      } else if (flag == "--min-os" && i + 1 < argc) {
        std::string value = argv[++i];
        auto dot = value.find('.');
        auto major = number(value.substr(0, dot)), minor = dot == value.npos ? 0 : number(value.substr(dot + 1));
        if (major > 0xffff || minor > 0xffff) throw std::runtime_error("OS version is too large");
        opts.minOsMajor = uint16_t(major);
        opts.minOsMinor = uint16_t(minor);
      } else if ((flag == "--buffer" || flag == "--texture" || flag == "--push-constant") && i + 1 < argc) {
        std::string value = argv[++i];
        if (flag == "--push-constant") {
          auto index = number(value);
          if (index > uint32_t(std::numeric_limits<int>::max())) throw std::runtime_error("push constant index is too large");
          opts.pushConstantBuffer = int(index);
        } else binding(opts, value, flag == "--texture");
      } else throw std::runtime_error("unknown option or missing argument: " + flag);
    }
    std::ifstream in(argv[1], std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open input");
    auto bytes = in.tellg();
    if (bytes < 20 || bytes % 4) throw std::runtime_error("input must contain whole SPIR-V words and a header");
    std::vector<uint32_t> words(size_t(bytes) / 4);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(words.data()), bytes)) throw std::runtime_error("cannot read input");
    auto result = spirv2air::translate(words.data(), words.size(), opts);
    if (!result.ok()) { std::fprintf(stderr, "translation failed: %s\n", result.error.c_str()); return 1; }
    std::ofstream out(argv[2], std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(result.metallib.data()), result.metallib.size())) throw std::runtime_error("cannot write output");
    out.close();
    if (!out) throw std::runtime_error("cannot finish writing output");
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 2; }
}
