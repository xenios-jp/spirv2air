// Parses synthetic containers, rebuilds them with mtlb::write, and requires byte identity.
// The parser here is test-only and assumes the tag set mtlb understands; unknown tags fail the test.
#include "metallib/metallib.h"
#include "airkit/airkit.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

template <class T>
T rd(const uint8_t* p) {
  T v;
  memcpy(&v, p, sizeof v);
  return v;
}

struct Parsed {
  mtlb::Target target;
  std::vector<mtlb::Function> fns;
  std::vector<std::vector<mtlb::FunctionConstant>> constants;
  std::vector<std::vector<mtlb::VertexAttribute>> attributes;
};

bool fail(const char* path, const char* why) {
  fprintf(stderr, "%s: %s\n", path, why);
  return false;
}

bool parse(const char* path, const std::vector<uint8_t>& d, Parsed& out) {
  if (d.size() < mtlb::kHeaderSize || memcmp(d.data(), "MTLB", 4)) return fail(path, "not a metallib");
  const uint8_t* h = d.data();
  out.target.platform = rd<uint16_t>(h + 4);
  out.target.file_version = {rd<uint16_t>(h + 6), rd<uint16_t>(h + 8)};
  out.target.file_type = h[10];
  out.target.os = h[11];
  out.target.os_version = {rd<uint16_t>(h + 12), rd<uint16_t>(h + 14)};
  const uint64_t fl_off = rd<uint64_t>(h + 0x18), pub_off = rd<uint64_t>(h + 0x28);
  const uint64_t bc_off = rd<uint64_t>(h + 0x48);

  const uint32_t count = rd<uint32_t>(h + fl_off);
  out.fns.resize(count);
  out.constants.resize(count);
  out.attributes.resize(count);
  const uint8_t* p = h + fl_off + 4;
  for (uint32_t i = 0; i < count; i++) {
    mtlb::Function& f = out.fns[i];
    const uint8_t* entry_end = p + rd<uint32_t>(p);
    const uint8_t* t = p + 4;
    uint64_t offt[3] = {};
    while (memcmp(t, "ENDT", 4)) {
      const uint16_t len = rd<uint16_t>(t + 4);
      const uint8_t* v = t + 6;
      if (!memcmp(t, "NAME", 4)) f.name = std::string_view(reinterpret_cast<const char*>(v), len - 1);
      else if (!memcmp(t, "TYPE", 4)) f.type = static_cast<mtlb::FunctionType>(v[0]);
      else if (!memcmp(t, "HASH", 4)) {}  // recomputed by the writer and compared via the bytes
      else if (!memcmp(t, "MDSZ", 4)) f.bitcode_size = rd<uint64_t>(v);
      else if (!memcmp(t, "OFFT", 4)) memcpy(offt, v, 24);
      else if (!memcmp(t, "VERS", 4)) {
        f.air = {rd<uint16_t>(v), rd<uint16_t>(v + 2)};
        f.language = {rd<uint16_t>(v + 4), rd<uint16_t>(v + 6)};
      } else if (!memcmp(t, "TESS", 4)) f.tessellation = v[0];
      else return fail(path, "unknown function tag");
      t = v + len;
    }
    if (t + 4 != entry_end) return fail(path, "entry size mismatch");
    f.bitcode = h + bc_off + offt[2];

    // Public metadata for this function.
    const uint8_t* m = h + pub_off + offt[0] + 4;
    while (memcmp(m, "ENDT", 4)) {
      const uint16_t len = rd<uint16_t>(m + 4);
      const uint8_t* v = m + 6;
      const uint16_t n = rd<uint16_t>(v);
      const uint8_t* q = v + 2;
      if (!memcmp(m, "CNST", 4)) {
        for (uint16_t k = 0; k < n; k++) {
          mtlb::FunctionConstant fc{};
          fc.name = reinterpret_cast<const char*>(q);
          q += fc.name.size() + 1;
          fc.type = q[0];
          fc.index = rd<uint16_t>(q + 1);
          fc.required = q[3];
          q += 4;
          out.constants[i].push_back(fc);
        }
      } else if (!memcmp(m, "VATT", 4)) {
        for (uint16_t k = 0; k < n; k++) {
          mtlb::VertexAttribute a{};
          a.name = reinterpret_cast<const char*>(q);
          q += a.name.size() + 1;
          const uint16_t packed = rd<uint16_t>(q);
          a.index = packed & 0x7fff;
          a.used = packed & 0x8000;
          q += 2;
          out.attributes[i].push_back(a);
        }
      } else if (!memcmp(m, "VATY", 4)) {
        for (uint16_t k = 0; k < n; k++) out.attributes[i][k].type = q[k];
      } else return fail(path, "unknown public metadata tag");
      m = v + len;
    }
    f.constants = out.constants[i].data();
    f.num_constants = static_cast<uint32_t>(out.constants[i].size());
    f.attributes = out.attributes[i].data();
    f.num_attributes = static_cast<uint32_t>(out.attributes[i].size());
    p = entry_end;
  }
  return true;
}

bool check(const char* path, const std::vector<uint8_t>& ref) {
  if (ref.empty()) return fail(path, "cannot read");
  Parsed parsed;
  if (!parse(path, ref, parsed)) return false;

  const size_t size = mtlb::library_size(parsed.fns.data(), parsed.fns.size());
  if (size != ref.size()) return fail(path, "library_size differs from reference");
  std::vector<uint8_t> out(size);
  mtlb::write(out.data(), parsed.target, parsed.fns.data(), parsed.fns.size());
  if (out != ref) {
    size_t i = 0;
    while (out[i] == ref[i]) i++;
    fprintf(stderr, "%s: first difference at byte 0x%zx\n", path, i);
    return false;
  }

  // Zero-copy path: bitcode written in place before write() must give the same bytes.
  std::vector<uint8_t> zc(size);
  uint8_t* dst = zc.data() + mtlb::prefix_size(parsed.fns.data(), parsed.fns.size());
  std::vector<mtlb::Function> fns = parsed.fns;
  for (mtlb::Function& f : fns) {
    memcpy(dst, f.bitcode, f.bitcode_size);
    f.bitcode = dst;
    dst += f.bitcode_size;
  }
  mtlb::write(zc.data(), parsed.target, fns.data(), fns.size());
  if (zc != ref) return fail(path, "zero-copy output differs");

  printf("ok  %s (%zu functions, %zu bytes)\n", path, parsed.fns.size(), size);
  return true;
}

}  // namespace

// Target expectations from Apple's toolchain: the triple in the module and the header fields written by `metallib`
// for MSL compiled at each deployment target (macosx, iphoneos and iphonesimulator SDKs).
bool checkTargets() {
  struct Case {
    mtlb::Platform platform;
    mtlb::Version os;
    const char* triple;
    uint16_t platform_field, file_minor;
    uint8_t os_byte;
    uint16_t air_minor;
  };
  const Case cases[] = {
      {mtlb::Platform::MacOS, {14, 0}, "air64_v26-apple-macosx14.0.0", 0x8001, 7, 0x81, 6},
      {mtlb::Platform::MacOS, {15, 0}, "air64_v27-apple-macosx15.0.0", 0x8001, 8, 0x81, 7},
      {mtlb::Platform::MacOS, {26, 0}, "air64_v28-apple-macosx26.0.0", 0x8001, 9, 0x81, 8},
      {mtlb::Platform::IOS, {18, 0}, "air64_v27-apple-ios18.0.0", 0x0001, 8, 0x82, 7},
      {mtlb::Platform::IOSSimulator, {18, 0}, "air64_v27-apple-ios18.0.0-simulator", 0x0001, 8, 0x87, 7},
      {mtlb::Platform::IOS, {26, 0}, "air64_v28-apple-ios26.0.0", 0x0001, 9, 0x82, 8},
      {mtlb::Platform::IOS, {26, 2}, "air64_v28-apple-ios26.2.0", 0x0001, 9, 0x82, 8},
      {mtlb::Platform::IOSSimulator, {26, 0}, "air64_v28-apple-ios26.0.0-simulator", 0x0001, 9, 0x87, 8},
  };
  bool ok = true;
  for (const Case& c : cases) {
    char buf[64];
    const mtlb::Target t = mtlb::make_target(c.platform, c.os);
    const bool good = !strcmp(mtlb::triple(c.platform, c.os, buf, sizeof buf), c.triple) &&
                      t.platform == c.platform_field && t.file_version.major == 2 &&
                      t.file_version.minor == c.file_minor && t.os == c.os_byte && t.os_version.major == c.os.major &&
                      t.os_version.minor == c.os.minor && t.file_type == 0 &&
                      mtlb::air_version(c.platform, c.os).minor == c.air_minor;
    if (!good) fprintf(stderr, "target mismatch: %s\n", c.triple);
    ok = ok && good;
  }
  // Defaults select the oldest supported version; older requests are raised to it.
  const mtlb::Target mac = mtlb::make_target(mtlb::Platform::MacOS),
                     ios = mtlb::make_target(mtlb::Platform::IOS, {17, 2});
  const mtlb::Target legacy;
  ok = ok && mac.platform == legacy.platform && mac.os == legacy.os && mac.os_version.major == 14 &&
       mac.file_version.minor == legacy.file_version.minor && ios.os_version.major == 18 && ios.os_version.minor == 0 &&
       mtlb::make_target(mtlb::Platform::IOS).os_version.major == 18;
  if (!ok) fprintf(stderr, "target defaults mismatch\n");
  return ok;
}

int main() {
  airkit::Kernel a("first"), b("second");
  a.begin().retVoid(); b.begin().retVoid();
  auto one = a.bitcode(), two = b.bitcode();
  auto lib = airkit::buildMetallib({{"first", one}, {"second", two}});
  bool ok = check("multi-function", lib);
  mtlb::FunctionConstant constant{"flag", 53, 2, 1};
  mtlb::VertexAttribute attribute{"position", 3, true, 6};
  mtlb::Function function;
  function.name = "vertex"; function.type = mtlb::FunctionType::Vertex;
  function.bitcode = one.data(); function.bitcode_size = one.size();
  function.constants = &constant; function.num_constants = 1;
  function.attributes = &attribute; function.num_attributes = 1;
  function.tessellation = mtlb::tessellation_tag(false, 3);
  std::vector<uint8_t> tagged(mtlb::library_size(&function, 1));
  mtlb::write(tagged.data(), {}, &function, 1);
  ok = check("tagged-function", tagged) && ok;
  ok = checkTargets() && ok;
  return ok ? 0 : 1;
}
