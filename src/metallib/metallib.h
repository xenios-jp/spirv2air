// metallib container writer.
//
// Layout (all little-endian), as produced by `xcrun metallib`:
//   header (0x58 bytes)
//   function list   : u32 count, then per function: u32 entry_size (includes itself) + tags + ENDT
//   header extension: tags + ENDT (empty here)
//   public metadata : per function: u32 tag_bytes (tags + ENDT) + tags + ENDT
//   private metadata: same shape (always empty here)
//   bitcode         : one wrapped AIR bitcode module per function, concatenated
// A tag is a 4-char code, u16 length, payload; ENDT has no length.
//
// Speed: the writer never allocates. Callers size the buffer with library_size(), or use the zero-copy
// path: write bitcode modules directly at out + prefix_size() (in function order, contiguous), point
// each Function::bitcode at its location, then call write(). Bitcode already in place is not copied.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mtlb {

enum class FunctionType : uint8_t {
  Vertex = 0,
  Fragment = 1,
  Kernel = 2,
  Visible = 4,
  Mesh = 7,
  Object = 8,
};

struct Version {
  uint16_t major, minor;
};

// CNST payload entry. type is an MTLDataType value (53 = bool).
struct FunctionConstant {
  std::string_view name;
  uint8_t type;
  uint16_t index;
  uint8_t required;  // observed 1 for a constant the function uses
};

// VATT/VATY payload entry. type is an MTLDataType value (6 = float4, 4 = float2, 32 = int4).
struct VertexAttribute {
  std::string_view name;
  uint16_t index;  // < 0x8000
  bool used;       // sets bit 15 of the packed index
  uint8_t type;
};

struct Function {
  std::string_view name;
  FunctionType type = FunctionType::Kernel;
  const uint8_t* bitcode = nullptr;  // wrapped AIR bitcode (0x0B17C0DE header)
  uint64_t bitcode_size = 0;
  // Optional precomputed HASH (32 bytes); null = SHA-256 of the bitcode (~1.2 us for a 3-4 KB module).
  // The driver does not validate HASH but keys its shader cache on it: two different modules with the
  // same HASH run whichever was compiled first (verified 2026-10-14). Only pass a value that is a
  // collision-resistant digest of everything that determines the bitcode bytes.
  const uint8_t* sha256 = nullptr;
  Version air{2, 6};                 // VERS: AIR version
  Version language{3, 1};            // VERS: Metal language version
  const FunctionConstant* constants = nullptr;
  uint32_t num_constants = 0;
  const VertexAttribute* attributes = nullptr;
  uint32_t num_attributes = 0;
  // TESS: set for a post-tessellation vertex function, to its patch type | control points << 2 (patch type 1 for
  // triangles, 2 for quads), as Apple's metallib linker writes it. 0 writes no TESS tag.
  uint8_t tessellation = 0;
};

// The TESS value of a post-tessellation vertex function.
constexpr uint8_t tessellation_tag(bool quad, unsigned control_points) {
  return static_cast<uint8_t>((quad ? 2 : 1) | control_points << 2);
}

// Target platform of a library. The values map to the header's OS byte (macOS 0x81, iOS 0x82, iOS Simulator 0x87).
enum class Platform : uint8_t { MacOS, IOS, IOSSimulator };

// Defaults target macOS 14, file version 2.7 and AIR 2.6: the header fields `xcrun metallib` writes for a library built
// with -mmacosx-version-min=14 (bytes 4-15 are 01 80 02 00 07 00 00 81 0e 00 00 00).
struct Target {
  uint16_t platform = 0x8001;  // bit 15 set; observed on macOS
  Version file_version{2, 7};
  uint8_t file_type = 0;       // 0 = executable library
  uint8_t os = 0x81;           // macOS
  Version os_version{14, 0};
};

// The target for a platform and minimum OS version, as Apple's metallib tool writes it for the same deployment
// target. A version of {0, 0} selects the oldest supported one: macOS 14 (the first with Metal 3.1) or iOS 18; older
// versions are raised to it, and iOS has no older paths. iOS and iOS Simulator headers have platform 0x0001 (macOS
// sets bit 15).
Target make_target(Platform platform, Version min_os = {0, 0});

// The AIR version Apple's compiler writes for a deployment target. macOS: 2.6 for 14, 2.7 for 15, 2.8 from 26. iOS and
// iOS Simulator: 2.7 for 18, 2.8 from 26 (later versions are written as iOS 26). The metallib file version is the same
// major with the minor one higher.
Version air_version(Platform platform, Version os);

// The module triple as Apple's compiler writes it for the deployment target: the arch carries the AIR version,
// "air64_v26-apple-macosx14.0.0", "air64_v27-apple-ios18.0.0[-simulator]".
const char* triple(Platform platform, Version os, char* buf, size_t size);

constexpr size_t kHeaderSize = 0x58;

// Bytes before the bitcode section. Depends only on names, types and metadata, not on bitcode.
size_t prefix_size(const Function* fns, size_t n);

// Total library size: prefix_size() + sum of bitcode sizes.
size_t library_size(const Function* fns, size_t n);

// Writes the complete library into out (library_size() bytes). Returns bytes written.
size_t write(uint8_t* out, const Target& target, const Function* fns, size_t n);

}  // namespace mtlb
