// spirv2air: SPIR-V -> Apple AIR -> metallib.
//
// Vertex, fragment and tessellation-evaluation modules use the graphics argument-buffer layout in Options.
// GLCompute modules use direct Metal resource bindings. The output entry point is named "main".
// Unsupported constructs return a diagnostic in Result::error. See the public README for the supported subset.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace spirv2air {

// Identifies the output format for caches: two builds with the same kVersion must produce byte-identical output
// for the same input. Bump it with every change that changes any output byte.
inline constexpr char kVersion[] = "spirv2air 1.0.5";

struct Result {
  std::vector<uint8_t> metallib;  // one function, "main"
  std::string error;              // empty on success
  bool ok() const { return error.empty(); }
};

// Where one SPIR-V (descriptor set, binding) goes in a direct-binding module (GLCompute). This has the shape of
// SPIRV-Cross's MSLResourceBinding: the kind of the variable at (set, binding) selects which index is used, so one
// entry can serve a buffer and a texture that share a (set, binding) across modules.
struct ResourceBinding {
  uint32_t set = 0, binding = 0;
  uint32_t buffer = 0;   // [[buffer(n)]]: storage buffers, and uniform buffers
  uint32_t texture = 0;  // [[texture(n)]]: sampled images
};

// Graphics resource layout. Defaults preserve the original xenia integration contract. This is a host ABI,
// not a general Vulkan descriptor mapper. All offsets/strides are bytes; pointer slots must be 8-byte aligned.
// Different bindings and descriptor sets must not alias resources that are used together.
// A device buffer a graphics shader reads and writes through a StorageBuffer variable at (set, binding): its
// address is the 8-byte pointer at byte argumentBufferOffset of the top-level argument buffer. Loads and integer
// atomics (add, and, or, min, max) are supported, in every graphics stage.
struct StorageBufferSlot {
  uint32_t set = 0, binding = 0, argumentBufferOffset = 0;
};

struct GraphicsBindings {
  uint32_t resourceHeapBuffer = 0, samplerHeapBuffer = 1, argumentBuffer = 2;
  // One float per control point, patch after patch. Negative disables this input (translation then fails if used).
  int controlPointBuffer = 6;
  // Fragment-stage coherent uint occlusion counter (ZPD in the original integration). Negative disables it.
  int occlusionCounterBuffer = 7;
  uint32_t sharedMemorySet = 0, sharedMemoryBinding = 0;
  uint32_t occlusionCounterSet = 0, occlusionCounterBinding = 1;
  // Fragment-stage raster-ordered storage buffer (xenia's EDRAM under fragment shader interlock): the
  // StorageBuffer at (rasterOrderedSet, rasterOrderedBinding) becomes this direct device buffer in raster order
  // group 0, so the fragments covering a pixel access it in primitive order. Pixel interlock, which it implements,
  // is rejected while this is negative (default).
  int rasterOrderedBuffer = -1;
  uint32_t rasterOrderedSet = 0, rasterOrderedBinding = 2;
  uint32_t constantsSet = 1, constantBufferCount = 5;
  // "Fetch state" uniforms as Metal function constants. A Uniform variable in this descriptor set must be a block
  // holding one uint4; binding N becomes the uint4 function constant named "spirv2air_fc_<N>" (function constant index
  // N), set by the host in MTLFunctionConstantValues (setConstantValue:type:withName:, type MTLDataTypeUInt4). Reads
  // of the variable (member 0, optionally one component) read the constant. The shader fails to specialize in Metal
  // until every constant it uses has a value. Negative disables this (default). The set takes precedence over
  // constantsSet.
  int32_t functionConstantSet = -1;
  // Further StorageBuffer variables, besides shared memory and the occlusion counter (xenia's tile cull bounds:
  // {0, 4, 112}). A StorageBuffer variable matching none of the entries is rejected.
  std::vector<StorageBufferSlot> storageBuffers;
  uint32_t vertexTexturesSet = 2, fragmentTexturesSet = 3;
  // Uniform binding n loads its pointer at constantBaseOffset + n * constantStride.
  uint32_t constantBaseOffset = 0, constantStride = 8;
  uint32_t sharedMemoryReadOffset = 48, sharedMemoryWriteOffset = 56;
  uint32_t vertexTextureIndicesOffset = 64, fragmentTextureIndicesOffset = 72;
  // Per-binding index records in the stage's index buffer contain two uint heap indices.
  uint32_t textureIndexStride = 8, textureIndexOffset = 0, samplerIndexOffset = 4;
  // Heap records hold 64-bit handles. Defaults: {buffer address, texture handle, metadata}.
  uint32_t descriptorStride = 24, textureHandleOffset = 8, samplerHandleOffset = 0;
};

// Target platform. The default, macOS 14, reproduces the original output byte for byte. iOS and iOS Simulator write
// the triple, AIR version and metallib header that Apple's toolchain writes for the same deployment target. The
// generated code is identical on all three; only the module triple, air.version and the container header differ.
enum class Platform { MacOS, IOS, IOSSimulator };

// Conversion policy for GLSL.std.450 PackHalf2x16. This does not affect
// QuantizeToF16 or other floating-point operations. Include it in cache keys.
enum class HalfPackingRounding { NearestEven, TowardZero };

struct Options {
  Platform platform = Platform::MacOS;
  // Minimum OS version (macOS or iOS, per platform). 0.0 selects the default: macOS 14 or iOS 18, which is also the
  // iOS floor; older versions are raised to it. The AIR version follows it as Apple's compiler does: macOS 2.6 up to
  // 14, 2.7 for 15, 2.8 from 26; iOS 2.7 for 18, 2.8 from 26.
  uint16_t minOsMajor = 0, minOsMinor = 0;
  // Used by GLCompute modules. A resource without an entry is an error.
  std::vector<ResourceBinding> bindings;
  // [[buffer(n)]] of the push constant block (a constant-address-space buffer; bind it with setBytes). -1: none.
  int pushConstantBuffer = -1;
  // Thread constant jumps through a switch-based dispatch loop (including the Xenos loop emitted by xenia).
  // Enabled by default; disabling preserves the original dispatch control flow. This option affects output bytes.
  bool jumpThread = true;
  // Used by vertex, fragment and tessellation-evaluation modules; ignored by GLCompute.
  GraphicsBindings graphics;
  // Original graphics conventions: VertexIndex = vertex_id - base_vertex; nonfinite position xyz becomes NaN.
  // Disable for callers that want the native Metal vertex_id and unmodified position respectively.
  bool vertexIndexRelativeToBase = true;
  bool positionInfToNan = true;
  HalfPackingRounding halfPackingRounding = HalfPackingRounding::NearestEven;
};

Result translate(const uint32_t* words, size_t word_count, const Options& options = {});

}  // namespace spirv2air
