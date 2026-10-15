// Checks the approximate math intrinsics spirv2air uses for pixel shaders (air.fast_fmin, fmax, sqrt, rint and so on)
// on the GPU with NaN, infinity and signed-zero operands. Calls carry no fast-math flags: nnan/ninf would let the
// compiler assume those values never occur, which SPIR-V allows only with the FPFastMathMode NotNaN/NotInf bits and
// nsz only with NSZ, and a run of every operand pair gave the same results with and without the flags on this GPU
// (nothing to gain from them). SPIR-V NMin/NMax return the other operand when one is NaN; the saturate/clamp pair
// and Sqrt/RoundEven follow IEEE (and D3D12 where SPIR-V leaves NaN open).
//
// The GLSL.std.450 math and geometric instructions of batch 1 (Round Tan Asin Acos Atan Atan2 Sinh Cosh Tanh Asinh
// Acosh Atanh Radians Degrees FMix Step SmoothStep Length Distance Cross Normalize FaceForward Reflect Refract) are
// checked against hand-written MSL for the same operations, compiled at run time, on the same inputs: compute (the MSL
// with fast math off) and fragment (default fast math), bit for bit with NaN == NaN, over ordinary values, zeros of
// both signs, infinities, NaNs and values that overflow or underflow a squared length.
// usage: math_gpu <ieee.frag.spv> <glsl_ext.comp.spv> <glsl_ext.frag.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr int kResourceHeap = 0, kSamplerHeap = 1, kTlab = 2;

std::vector<uint32_t> readWords(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(f ? size_t(f.tellg()) / 4 : 0);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
  return words;
}

NSString* const kSource = @R"(
#include <metal_stdlib>
using namespace metal;
vertex float4 vtx(uint v [[vertex_id]]) {
  const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  return float4(p[v], 0.5, 1);
}
)";

NSString* const kCompute = @R"(
#include <metal_stdlib>
using namespace metal;
inline void put(device float* o, thread uint& k, float v) { o[k++] = v; }
inline void put(device float* o, thread uint& k, float2 v) { put(o, k, v.x); put(o, k, v.y); }
inline void put(device float* o, thread uint& k, float3 v) { put(o, k, v.xy); put(o, k, v.z); }
inline void put(device float* o, thread uint& k, float4 v) { put(o, k, v.xyz); put(o, k, v.w); }
// SPIR-V Step: 0 if x < edge, otherwise 1, so a NaN operand gives 1 (MSL step() gives 0).
template <typename T> T spvStep(T edge, T x) { return select(T(1), T(0), x < edge); }
template <typename T> void mathOps(T a, T b, T c, device float* o, thread uint& k) {
  put(o, k, round(a)); put(o, k, tan(a)); put(o, k, asin(a)); put(o, k, acos(a)); put(o, k, atan(a));
  put(o, k, atan2(a, b)); put(o, k, sinh(a)); put(o, k, cosh(a)); put(o, k, tanh(a)); put(o, k, asinh(a));
  put(o, k, acosh(a)); put(o, k, atanh(a)); put(o, k, a * (M_PI_F / 180.0f)); put(o, k, a * (180.0f / M_PI_F));
  put(o, k, mix(a, b, c)); put(o, k, spvStep(a, b)); put(o, k, smoothstep(a, b, c));
}
template <typename T> void geoOps(T a, T b, T c, float eta, device float* o, thread uint& k) {
  put(o, k, length(a)); put(o, k, distance(a, b)); put(o, k, normalize(a)); put(o, k, faceforward(a, b, c));
  put(o, k, reflect(a, b)); put(o, k, refract(a, b, eta));
}
kernel void ref(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                uint gid [[thread_position_in_grid]]) {
  device const float* p = in + gid * 12;
  device float* o = out + gid * 221;
  uint k = 0;
  float4 a = float4(p[0], p[1], p[2], p[3]), b = float4(p[4], p[5], p[6], p[7]), c = float4(p[8], p[9], p[10], p[11]);
  mathOps(a.x, b.x, c.x, o, k);
  mathOps(a.xy, b.xy, c.xy, o, k);
  mathOps(a.xyz, b.xyz, c.xyz, o, k);
  mathOps(a, b, c, o, k);
  geoOps(a.xy, b.xy, c.xy, c.x, o, k);
  geoOps(a.xyz, b.xyz, c.xyz, c.x, o, k);
  geoOps(a, b, c, c.x, o, k);
  put(o, k, cross(a.xyz, b.xyz));
  // MSL has no scalar geometric functions: Length and Distance are |x|, the others use a vector with a zero y.
  float2 za = float2(a.x, 0), zb = float2(b.x, 0), zc = float2(c.x, 0);
  put(o, k, fabs(a.x)); put(o, k, fabs(a.x - b.x)); put(o, k, normalize(za).x); put(o, k, faceforward(za, zb, zc).x);
  put(o, k, reflect(za, zb).x); put(o, k, refract(za, zb, c.x).x);
}
)";

NSString* const kFragment = @R"(
#include <metal_stdlib>
using namespace metal;
// SPIR-V Step: 0 if x < edge, otherwise 1, so a NaN operand gives 1 (MSL step() gives 0).
template <typename T> T spvStep(T edge, T x) { return select(T(1), T(0), x < edge); }
float4 expected(uint x, float4 a, float4 b, float4 c) {
  float sa = a.x, sb = b.x, sc = c.x;
  switch (x) {
    case 0: return round(a);
    case 1: return tan(a);
    case 2: return asin(a);
    case 3: return acos(a);
    case 4: return atan(a);
    case 5: return atan2(a, b);
    case 6: return sinh(a);
    case 7: return cosh(a);
    case 8: return tanh(a);
    case 9: return asinh(a);
    case 10: return acosh(a);
    case 11: return atanh(a);
    case 12: return a * (M_PI_F / 180.0f);
    case 13: return a * (180.0f / M_PI_F);
    case 14: return mix(a, b, c);
    case 15: return spvStep(a, b);
    case 16: return smoothstep(a, b, c);
    case 17: return float4(round(sa), tan(sa), asin(sa), acos(sa));
    case 18: return float4(atan(sa), atan2(sa, sb), sinh(sa), cosh(sa));
    case 19: return float4(tanh(sa), asinh(sa), acosh(sa), atanh(sa));
    case 20: return float4(sa * (M_PI_F / 180.0f), sa * (180.0f / M_PI_F), mix(sa, sb, sc), spvStep(sa, sb));
    case 21: return float4(smoothstep(sa, sb, sc), 0, 0, 0);
    case 22: return float4(length(a.xy), 0, 0, 0);
    case 23: return float4(distance(a.xy, b.xy), 0, 0, 0);
    case 24: return float4(normalize(a.xy), 0, 0);
    case 25: return float4(faceforward(a.xy, b.xy, c.xy), 0, 0);
    case 26: return float4(reflect(a.xy, b.xy), 0, 0);
    case 27: return float4(refract(a.xy, b.xy, sc), 0, 0);
    case 28: return float4(length(a.xyz), 0, 0, 0);
    case 29: return float4(distance(a.xyz, b.xyz), 0, 0, 0);
    case 30: return float4(normalize(a.xyz), 0);
    case 31: return float4(faceforward(a.xyz, b.xyz, c.xyz), 0);
    case 32: return float4(reflect(a.xyz, b.xyz), 0);
    case 33: return float4(refract(a.xyz, b.xyz, sc), 0);
    case 34: return float4(length(a), 0, 0, 0);
    case 35: return float4(distance(a, b), 0, 0, 0);
    case 36: return normalize(a);
    case 37: return faceforward(a, b, c);
    case 38: return reflect(a, b);
    case 39: return refract(a, b, sc);
    case 40: return float4(cross(a.xyz, b.xyz), 0);
    default: return float4(0);
  }
}
fragment float4 ref(float4 pos [[position]], constant float4* rows [[buffer(0)]]) {
  uint y = uint(pos.y);
  return expected(uint(pos.x), rows[3 * y], rows[3 * y + 1], rows[3 * y + 2]);
}
)";

bool sameFloat(float got, float want, bool signedZeroMatters) {
  if (std::isnan(want)) return std::isnan(got);
  if (want == 0 && got == 0 && !signedZeroMatters) return true;
  return memcmp(&got, &want, 4) == 0;
}

int testIeee(id<MTLDevice> device, id<MTLCommandQueue> queue, const char* path) {
  @autoreleasepool {
    NSError* error = nil;
    id<MTLLibrary> source = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!source) return fprintf(stderr, "MSL: %s\n", error.description.UTF8String), 1;
    std::vector<uint32_t> words = readWords(path);
    spirv2air::Result r = spirv2air::translate(words.data(), words.size());
    if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), 1;
    dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> lib = [device newLibraryWithData:blob error:&error];
    if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), 1;
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [source newFunctionWithName:@"vtx"];
    d.fragmentFunction = [lib newFunctionWithName:@"main"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    id<MTLRenderPipelineState> state = [device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) return fprintf(stderr, "pipeline: %s\n", error.description.UTF8String), 1;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:2
                                                                                 height:2
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    id<MTLBuffer> resourceHeap = [device newBufferWithLength:24 options:MTLResourceStorageModeShared];
    id<MTLBuffer> samplerHeap = [device newBufferWithLength:24 options:MTLResourceStorageModeShared];
    id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];
    memset(tlab.contents, 0, 112);

    const float nan = std::nanf(""), inf = INFINITY;
    const float values[] = {nan, inf, -inf, 0.0f, -0.0f, 1.0f, -1.0f, 2.5f, -2.5f, 0.5f, -0.5f, 1e-30f, 4.0f};
    int failures = 0, cases = 0;
    for (float a : values) {
      for (float b : values) {
        float coord[4] = {a, b, 0, 0}, got[4];
        id<MTLBuffer> ubo = [device newBufferWithBytes:coord length:16 options:MTLResourceStorageModeShared];
        uint64_t address = ubo.gpuAddress;
        memcpy(tlab.contents, &address, 8);
        id<MTLTexture> target = [device newTextureWithDescriptor:td];
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = target;
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(-7, -7, -7, -7);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
        [enc setRenderPipelineState:state];
        [enc setFragmentBuffer:resourceHeap offset:0 atIndex:kResourceHeap];
        [enc setFragmentBuffer:samplerHeap offset:0 atIndex:kSamplerHeap];
        [enc setFragmentBuffer:tlab offset:0 atIndex:kTlab];
        [enc useResource:ubo usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), 1;
        [target getBytes:got bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
        auto fmin = [](float x, float y) { return std::isnan(x) ? y : std::isnan(y) ? x : std::fmin(x, y); };
        auto fmax = [](float x, float y) { return std::isnan(x) ? y : std::isnan(y) ? x : std::fmax(x, y); };
        float rint = std::nearbyint(a);
        float want[4] = {fmin(a, b), fmax(a, b), fmin(fmax(a, 0.0f), 1.0f), std::sqrt(a) + (rint - rint)};
        for (int k = 0; k < 4; ++k) {
          ++cases;
          if (!sameFloat(got[k], want[k], k == 3)) {
            if (++failures <= 20)
              fprintf(stderr, "FAIL a=%g b=%g component %d: got %g (0x%08x), want %g\n", a, b, k, got[k],
                      *reinterpret_cast<uint32_t*>(&got[k]), want[k]);
          }
        }
      }
    }
    printf("math_gpu: %d results, %d failures\n", cases, failures);
    return failures ? 1 : 0;
  }
}

// A float source of inputs: 4 slots per vec4, rows of 12 (a, b, c).
std::vector<float> makeRows(unsigned rows) {
  const float nan = std::nanf(""), inf = INFINITY;
  const float tame[] = {0.0f,  -0.0f,  1.0f, -1.0f, 0.5f,       -0.5f, 2.5f,  -2.5f,
                        0.25f, -0.75f, 3.0f, -3.0f, 1.5707964f, 0.1f,  7.25f, -0.3f};
  const float wide[] = {1e30f, -1e30f, 1e-30f,  -1e-30f, 3e38f,     -3e38f, inf,  -inf,
                        nan,   100.0f, -100.0f, 1e-20f,  0x1p-149f, 1.0f,   2.0f, -0.0f};
  std::vector<float> v(size_t(rows) * 12, 0.0f);
  auto set = [&](unsigned row, std::initializer_list<float> x) {
    unsigned k = 0;
    for (float f : x) v[row * 12 + k++] = f;
  };
  set(1, {1, 0, 0, 0, 0, 1, 0, 0, 2.5f, 0, 0, 0});  // refract with k < 0
  set(2, {1e30f, 1e30f, 1e30f, 1e30f, -1e30f, 1e30f, 0, 1, 1.5f, 0, 0, 0});
  set(3, {1e-30f, -1e-30f, 1e-30f, 1e-30f, 1e-30f, 1e-30f, 0, 0, 0.5f, 0, 0, 0});
  set(4, {INFINITY, 0, 0, 0, 1, 0, 0, 0, 1.5f, 0, 0, 0});
  set(5, {std::nanf(""), 1, 1, 1, 1, 1, 1, 1, 0.5f, 0, 0, 0});
  set(6, {-0.0f, -0.0f, -0.0f, -0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0, 0, 0});
  set(7, {3e38f, 3e38f, 3e38f, 3e38f, 1, 2, 3, 4, 1.5f, 0, 0, 0});
  set(8, {1, 0.5f, -0.5f, 0, 1, 0.5f, -0.5f, 0, 1, 0.5f, -0.5f, 0});  // a == b == c
  set(9, {0.6f, 0.8f, 0, 0, 0, 1, 0, 0, 0.7f, 0, 0, 0});
  set(10, {-1, 1, 0.5f, 0.25f, 0, 0, 0, 0, 1, 1, 1, 1});
  set(11, {1, 2, 3, 4, 4, 3, 2, 1, 1.33f, -1, 1, 0});
  uint32_t state = 12345;
  auto next = [&]() {
    state = state * 1664525u + 1013904223u;
    return state >> 16;
  };
  for (unsigned row = 12; row < rows; ++row)
    for (unsigned k = 0; k < 12; ++k) {
      bool useWide = (row & 1) && next() % 3 == 0;
      v[row * 12 + k] = useWide ? wide[next() % 16] : tame[next() % 16];
    }
  return v;
}

bool sameBits(float got, float want) { return sameFloat(got, want, true); }

int report(const char* test, unsigned row, unsigned col, const float* in, float got, float want, int& failures) {
  if (++failures <= 20) {
    fprintf(stderr,
            "FAIL %s row %u column %u: got %g (0x%08x), want %g (0x%08x); a=(%g %g %g %g) b=(%g %g %g %g) c=(%g %g)\n",
            test, row, col, got, *reinterpret_cast<uint32_t*>(&got), want, *reinterpret_cast<uint32_t*>(&want), in[0],
            in[1], in[2], in[3], in[4], in[5], in[6], in[7], in[8], in[9]);
  }
  return 0;
}

int testGlslCompute(id<MTLDevice> device, id<MTLCommandQueue> queue, const char* path) {
  constexpr unsigned kRows = 128, kStride = 221;
  NSError* error = nil;
  MTLCompileOptions* options = [MTLCompileOptions new];
  if (@available(macOS 15.0, *)) {
    // The same as the offline compiler's -fno-fast-math: no fast-math flags and the precise math functions.
    options.mathMode = MTLMathModeSafe;
    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
  } else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
  id<MTLLibrary> source = [device newLibraryWithSource:kCompute options:options error:&error];
  if (!source) return fprintf(stderr, "MSL compute: %s\n", error.description.UTF8String), 1;
  id<MTLComputePipelineState> reference =
      [device newComputePipelineStateWithFunction:[source newFunctionWithName:@"ref"] error:&error];
  if (!reference) return fprintf(stderr, "reference pipeline: %s\n", error.description.UTF8String), 1;
  std::vector<uint32_t> words = readWords(path);
  spirv2air::Options translateOptions;
  translateOptions.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  spirv2air::Result r = spirv2air::translate(words.data(), words.size(), translateOptions);
  if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), 1;
  dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
  id<MTLLibrary> lib = [device newLibraryWithData:blob error:&error];
  if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), 1;
  id<MTLComputePipelineState> translated = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"main"]
                                                                                 error:&error];
  if (!translated) return fprintf(stderr, "pipeline: %s\n", error.description.UTF8String), 1;
  std::vector<float> input = makeRows(kRows);
  id<MTLBuffer> in = [device newBufferWithBytes:input.data()
                                         length:input.size() * 4
                                        options:MTLResourceStorageModeShared];
  id<MTLBuffer> outTranslated = [device newBufferWithLength:kRows * kStride * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> outReference = [device newBufferWithLength:kRows * kStride * 4 options:MTLResourceStorageModeShared];
  memset(outTranslated.contents, 0xFF, kRows * kStride * 4);
  memset(outReference.contents, 0xFF, kRows * kStride * 4);
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  for (int which = 0; which < 2; ++which) {
    [e setComputePipelineState:which ? reference : translated];
    [e setBuffer:in offset:0 atIndex:0];
    [e setBuffer:which ? outReference : outTranslated offset:0 atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(kRows / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  }
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), 1;
  const float *got = static_cast<const float*>(outTranslated.contents),
              *want = static_cast<const float*>(outReference.contents);
  int failures = 0;
  for (unsigned row = 0; row < kRows; ++row)
    for (unsigned col = 0; col < kStride; ++col)
      // Scalar Normalize..Refract (columns 217-220) are checked through a vector with a zero y lane, whose dot product
      // can differ from the scalar product in the sign of a zero; every other result must match bit for bit.
      if (!sameFloat(got[row * kStride + col], want[row * kStride + col], col < 217) &&
          !(std::isnan(got[row * kStride + col]) && std::isnan(want[row * kStride + col])))
        report("glsl compute", row, col, &input[row * 12], got[row * kStride + col], want[row * kStride + col],
               failures);
  printf("math_gpu glsl compute: %u results, %d failures\n", kRows * kStride, failures);
  return failures ? 1 : 0;
}

int testGlslFragment(id<MTLDevice> device, id<MTLCommandQueue> queue, const char* path) {
  constexpr unsigned kRows = 64, kCases = 41;
  NSError* error = nil;
  id<MTLLibrary> vertexLib = [device newLibraryWithSource:kSource options:nil error:&error];
  id<MTLLibrary> referenceLib = [device newLibraryWithSource:kFragment options:nil error:&error];
  if (!vertexLib || !referenceLib) return fprintf(stderr, "MSL fragment: %s\n", error.description.UTF8String), 1;
  std::vector<uint32_t> words = readWords(path);
  spirv2air::Result r = spirv2air::translate(words.data(), words.size());
  if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), 1;
  dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
  id<MTLLibrary> lib = [device newLibraryWithData:blob error:&error];
  if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), 1;
  auto pipeline = [&](id<MTLFunction> fragment) {
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [vertexLib newFunctionWithName:@"vtx"];
    d.fragmentFunction = fragment;
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
    return [device newRenderPipelineStateWithDescriptor:d error:&error];
  };
  id<MTLRenderPipelineState> translated = pipeline([lib newFunctionWithName:@"main"]);
  id<MTLRenderPipelineState> reference = pipeline([referenceLib newFunctionWithName:@"ref"]);
  if (!translated || !reference) return fprintf(stderr, "pipeline: %s\n", error.description.UTF8String), 1;
  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                width:kCases
                                                                               height:kRows
                                                                            mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget;
  td.storageMode = MTLStorageModeShared;
  std::vector<float> input = makeRows(kRows);
  id<MTLBuffer> ubo = [device newBufferWithBytes:input.data()
                                          length:input.size() * 4
                                         options:MTLResourceStorageModeShared];
  id<MTLBuffer> resourceHeap = [device newBufferWithLength:24 options:MTLResourceStorageModeShared];
  id<MTLBuffer> samplerHeap = [device newBufferWithLength:24 options:MTLResourceStorageModeShared];
  id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];
  memset(tlab.contents, 0, 112);
  uint64_t address = ubo.gpuAddress;
  memcpy(tlab.contents, &address, 8);
  std::vector<float> result[2];
  for (int which = 0; which < 2; ++which) {
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].clearColor = MTLClearColorMake(-7, -7, -7, -7);
    pass.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    [enc setRenderPipelineState:which ? reference : translated];
    if (which == 0) {
      [enc setFragmentBuffer:resourceHeap offset:0 atIndex:kResourceHeap];
      [enc setFragmentBuffer:samplerHeap offset:0 atIndex:kSamplerHeap];
      [enc setFragmentBuffer:tlab offset:0 atIndex:kTlab];
    } else {
      [enc setFragmentBuffer:ubo offset:0 atIndex:0];
    }
    [enc useResource:ubo usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), 1;
    result[which].resize(kRows * kCases * 4);
    [target getBytes:result[which].data()
         bytesPerRow:kCases * 16
          fromRegion:MTLRegionMake2D(0, 0, kCases, kRows)
         mipmapLevel:0];
  }
  int failures = 0;
  for (unsigned row = 0; row < kRows; ++row)
    for (unsigned x = 0; x < kCases * 4; ++x) {
      float g = result[0][row * kCases * 4 + x], w = result[1][row * kCases * 4 + x];
      unsigned kind = x / 4;
      bool refract = kind == 27 || kind == 33 || kind == 39;
      if (refract) {
        // Apple's fast-math Refract is free to reassociate and contract (nnan, ninf, reassoc, arcp, contract): the
        // compiled expression is not reproducible bit for bit, so allow a small absolute difference on ordinary inputs
        // and skip the rest.
        bool ordinary = true;
        for (unsigned k = 0; k < 12; ++k) {
          float v = std::fabs(input[row * 12 + k]);
          ordinary = ordinary && (v == 0 || (v >= 1e-6f && v <= 1e6f));
        }
        if (!ordinary) continue;
        // Absolute tolerance scaled by the magnitudes (the result can be a small difference of large terms).
        float scale = 1;
        for (unsigned k = 0; k < 8; ++k) scale = std::max(scale, std::fabs(input[row * 12 + k]));
        if (std::isfinite(g) && std::isfinite(w) &&
            std::fabs(g - w) <= 1e-6f * scale * scale * (1 + std::fabs(input[row * 12 + 8])))
          continue;
      }
      if (!sameBits(g, w)) report("glsl fragment", row, x, &input[row * 12], g, w, failures);
    }
  printf("math_gpu glsl fragment: %u results, %d failures\n", kRows * kCases * 4, failures);
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 4)
      return fprintf(stderr, "usage: math_gpu <ieee.frag.spv> <glsl_ext.comp.spv> <glsl_ext.frag.spv>\n"), 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    int status = testIeee(device, queue, argv[1]);
    status |= testGlslCompute(device, queue, argv[2]);
    status |= testGlslFragment(device, queue, argv[3]);
    return status;
  }
}
