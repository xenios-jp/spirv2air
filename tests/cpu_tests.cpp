// CPU translation and configuration checks over hand-written SPIR-V fixtures.
#include <spirv2air.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int failures;
static void check(bool ok, const std::string& message) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message.c_str()); ++failures; }
}
static std::vector<uint32_t> read(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() < 20 || f.tellg() % 4) return {};
  std::vector<uint32_t> w(size_t(f.tellg()) / 4);
  f.seekg(0); f.read(reinterpret_cast<char*>(w.data()), w.size() * 4);
  return f ? w : std::vector<uint32_t>{};
}
static spirv2air::Result run(const std::vector<uint32_t>& w, const spirv2air::Options& o = {}) {
  return spirv2air::translate(w.data(), w.size(), o);
}
static void differs(const std::vector<uint32_t>& w, const spirv2air::Options& a,
                    const spirv2air::Options& b, const char* message) {
  auto x = run(w, a), y = run(w, b);
  check(x.ok() && y.ok() && x.metallib != y.metallib, message);
}
static std::string hex(const uint8_t* p, size_t n) {
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    char b[3];
    std::snprintf(b, sizeof b, "%02x", p[i]);
    s += b;
  }
  return s;
}
// Header bytes 4..15 (platform, file version, file type, OS byte, OS version) and the VERS payload (AIR and language
// version) of a one-function library.
static std::string headerOf(const spirv2air::Result& r) { return hex(r.metallib.data() + 4, 12); }
static std::string versOf(const spirv2air::Result& r) {
  for (size_t i = 0x58; i + 14 < r.metallib.size() && i < 0x200; ++i)
    if (!std::memcmp(&r.metallib[i], "VERS", 4)) return hex(&r.metallib[i + 6], 8);
  return "";
}
// Expectations from Apple's toolchain (Metal 3.1 MSL compiled for each deployment target, then `metallib`):
// platform u16, file version 2.x, file type 0, OS byte (macOS 0x81, iOS 0x82, iOS Simulator 0x87), OS version u16 x2.
// VERS holds the AIR version (macOS 2.6/2.7/2.8, iOS 2.7/2.8) and Metal 3.1.
struct HeaderExpectation {
  spirv2air::Platform platform;
  uint16_t major, minor;
  const char* header;
  const char* vers;
};
static const HeaderExpectation kHeaders[] = {
    {spirv2air::Platform::MacOS, 0, 0, "01800200070000810e000000", "0200060003000100"},
    {spirv2air::Platform::MacOS, 15, 0, "01800200080000810f000000", "0200070003000100"},
    {spirv2air::Platform::MacOS, 26, 0, "01800200090000811a000000", "0200080003000100"},
    {spirv2air::Platform::IOS, 0, 0, "010002000800008212000000", "0200070003000100"},
    {spirv2air::Platform::IOS, 17, 0, "010002000800008212000000", "0200070003000100"},  // raised to the iOS 18 floor
    {spirv2air::Platform::IOS, 18, 0, "010002000800008212000000", "0200070003000100"},
    {spirv2air::Platform::IOS, 26, 0, "01000200090000821a000000", "0200080003000100"},
    {spirv2air::Platform::IOS, 26, 2, "01000200090000821a000200", "0200080003000100"},
    {spirv2air::Platform::IOSSimulator, 0, 0, "010002000800008712000000", "0200070003000100"},
    {spirv2air::Platform::IOSSimulator, 26, 0, "01000200090000871a000000", "0200080003000100"},
};
int main(int argc, char** argv) {
  if (argc != 27) return 2;  // 25 fixtures that translate, then the integer-input-without-Flat rejection fixture
  std::vector<std::vector<uint32_t>> fixtures;
  for (int i = 1; i < argc - 1; ++i) {
    auto w = read(argv[i]); check(!w.empty(), "read fixture");
    fixtures.push_back(w);
    spirv2air::Options o;
    if (i <= 6) {
      o.bindings = {{0,0,0,0}, {0,1,1,0}, {0,2,2,0}, {1,0,0,0}, {1,1,0,1}, {1,2,0,2}};
      o.pushConstantBuffer = 3;
    }
    auto result = run(w, o);
    check(result.ok(), std::string(argv[i]) + ": " + result.error);
    check(result.metallib.size() > 88 && !std::memcmp(result.metallib.data(), "MTLB", 4), "container header");
    check(result.metallib == run(w, o).metallib, "deterministic translation");
    if (i > 6) check(run(w).metallib == result.metallib, "explicit defaults equal implicit defaults");
  }
  auto noFlat = run(read(argv[argc - 1]));
  check(!noFlat.ok() && noFlat.error.find("without Flat") != std::string::npos,
        "reject integer fragment input without Flat");
  check(!run({}).ok(), "reject empty input");
  check(!run({0,0,0,0,0}).ok(), "reject invalid magic");
  auto bad = fixtures[0];
  bad.push_back(0); check(!run(bad).ok(), "reject zero-length instruction");
  bad = fixtures[0];
  bad.insert(bad.begin()+5, {uint32_t((2u << 16) | 17u), 11u}); // Capability Int64.
  check(run(bad).error.find("capability") != std::string::npos, "reject unsupported capability");
  check(!run(fixtures[0]).ok(), "require compute binding map");
  spirv2air::Options a, b;
  a.bindings = {{0,0,0,0}}; b = a; b.bindings[0].buffer = 4;
  differs(fixtures[0], a, b, "compute buffer relocation");
  a.bindings = {{0,0,0,0},{0,1,1,0}}; b = a; b.jumpThread = false;
  differs(fixtures[5], a, b, "dispatch threading changes output");
  a = {}; b = a; b.graphics.controlPointBuffer = 9;
  differs(fixtures[6], a, b, "control point buffer relocation");
  b.graphics.controlPointBuffer = -1; check(!run(fixtures[6], b).ok(), "disable control points");
  b = a; b.graphics.occlusionCounterBuffer = 9;
  differs(fixtures[9], a, b, "occlusion counter relocation");
  b.graphics.occlusionCounterBuffer = -1; check(!run(fixtures[9], b).ok(), "disable occlusion counter");
  b = a; b.graphics.argumentBuffer = 8;
  differs(fixtures[8], a, b, "argument buffer relocation");
  b = a; b.graphics.constantBaseOffset = 80;
  differs(fixtures[8], a, b, "constant pointer relocation");
  b = a; b.vertexIndexRelativeToBase = false;
  differs(fixtures[8], a, b, "native Metal vertex index");
  b = a; b.positionInfToNan = false;
  differs(fixtures[8], a, b, "unmodified vertex position");
  // Move the counter's SPIR-V descriptor set/binding together with its configured mapping. Metal bytes stay identical.
  auto moved = fixtures[9];
  // OpDecorate = 71; DescriptorSet = 34; Binding = 33. Identify the counter by its original binding 1.
  uint32_t counter = 0;
  for (size_t p=5; p<moved.size(); p+=moved[p]>>16)
    if ((moved[p]&65535u)==71 && moved[p+2]==33 && moved[p+3]==1) counter=moved[p+1];
  for (size_t p=5; p<moved.size(); p+=moved[p]>>16)
    if ((moved[p]&65535u)==71 && moved[p+1]==counter) {
      if (moved[p+2]==34) moved[p+3]=5;
      if (moved[p+2]==33) moved[p+3]=3;
    }
  b = a; b.graphics.occlusionCounterSet=5; b.graphics.occlusionCounterBinding=3;
  auto remapped = run(moved,b), original = run(fixtures[9]);
  check(counter && remapped.ok() && remapped.metallib==original.metallib, "descriptor remapping preserves Metal bytes");
  // Platform targets: header, file version and VERS follow the deployment target for every stage; the generated
  // code is the same, so only the container and module metadata differ. Fixtures: compute, post-tessellation
  // vertex, vertex, fragment.
  {
    const size_t stages[] = {0, 6, 8, 9};
    for (size_t index : stages) {
      spirv2air::Options base;
      if (index == 0) base.bindings = {{0, 0, 0, 0}};
      auto reference = run(fixtures[index], base);
      check(reference.ok() && headerOf(reference) == kHeaders[0].header && versOf(reference) == kHeaders[0].vers,
            "default target is macOS 14");
      for (const auto& e : kHeaders) {
        spirv2air::Options o = base;
        o.platform = e.platform;
        o.minOsMajor = e.major;
        o.minOsMinor = e.minor;
        auto r = run(fixtures[index], o);
        std::string what = "fixture " + std::to_string(index) + " platform " + std::to_string(int(e.platform)) +
                           " os " + std::to_string(e.major) + "." + std::to_string(e.minor);
        check(r.ok(), what + ": " + r.error);
        if (!r.ok()) continue;
        check(headerOf(r) == e.header, what + ": header " + headerOf(r));
        check(versOf(r) == e.vers, what + ": VERS " + versOf(r));
        check(run(fixtures[index], o).metallib == r.metallib, what + ": deterministic");
      }
      spirv2air::Options mac14 = base;
      mac14.minOsMajor = 14;
      check(run(fixtures[index], mac14).metallib == reference.metallib, "explicit macOS 14.0 equals the default");
      spirv2air::Options ios = base;
      ios.platform = spirv2air::Platform::IOS;
      auto iosResult = run(fixtures[index], ios);
      check(iosResult.ok() && iosResult.metallib != reference.metallib, "iOS output differs from macOS");
    }
  }
  std::printf("cpu_tests: %d failures\n", failures);
  return failures ? 1 : 0;
}
