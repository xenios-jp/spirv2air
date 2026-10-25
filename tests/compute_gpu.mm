// GLCompute GPU checks: invocation IDs, buffers, barriers, integer/float corner cases, PackHalf2x16, NoContraction
// products, normalized packing, relaxed atomics and image fetch/query operations. Inputs are our own SPIR-V assembly
// fixtures.
// usage: compute_gpu ids.spv shared.spv atomics.spv fetch.spv bits.spv ieee.spv pack.spv strictmul.spv integer.spv
//                    float_math.spv packing.spv atomic_extra.spv
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* test, const char* what, unsigned index, unsigned got, unsigned want) {
  if (ok) return;
  if (++failures <= 20) fprintf(stderr, "FAIL %s %s [%u]: 0x%08x, expected 0x%08x\n", test, what, index, got, want);
}

std::vector<uint32_t> readWords(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(f ? size_t(f.tellg()) / 4 : 0);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
  return words;
}

struct Context {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
};

id<MTLComputePipelineState> makePipeline(const Context& c, const char* test, const char* path,
                                         const spirv2air::Options& options, NSUInteger localThreads) {
  std::vector<uint32_t> words = readWords(path);
  spirv2air::Result r = spirv2air::translate(words.data(), words.size(), options);
  if (!r.ok()) return fprintf(stderr, "%s: translate failed: %s\n", test, r.error.c_str()), nil;
  NSError* error = nil;
  dispatch_data_t data = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
  id<MTLLibrary> library = [c.device newLibraryWithData:data error:&error];
  if (!library) return fprintf(stderr, "%s: newLibraryWithData: %s\n", test, error.description.UTF8String), nil;
  id<MTLFunction> function = [library newFunctionWithName:@"main"];
  if (!function || function.functionType != MTLFunctionTypeKernel)
    return fprintf(stderr, "%s: no kernel main\n", test), nil;
  id<MTLComputePipelineState> pipeline = [c.device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline) return fprintf(stderr, "%s: pipeline: %s\n", test, error.description.UTF8String), nil;
  if (pipeline.maxTotalThreadsPerThreadgroup < localThreads)
    return fprintf(stderr, "%s: max %lu threads per group, need %lu\n", test,
                   (unsigned long)pipeline.maxTotalThreadsPerThreadgroup, (unsigned long)localThreads),
           nil;
  return pipeline;
}

id<MTLBuffer> makeBuffer(const Context& c, const void* bytes, size_t length) {
  return [c.device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
}

bool wait(id<MTLCommandBuffer> cb, const char* test) {
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted)
    return fprintf(stderr, "%s: command buffer failed: %s\n", test, cb.error.description.UTF8String), false;
  return true;
}

bool testIds(const Context& c, const char* path) {
  const char* test = "ids";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 32);
  if (!pso) return false;
  const unsigned gx = 3, gy = 2, width = gx * 8, height = gy * 4, count = width * height * 2 * 4;
  std::vector<uint32_t> zeros(count, 0xFFFFFFFFu);
  id<MTLBuffer> out = makeBuffer(c, zeros.data(), count * 4);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:out offset:0 atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* v = static_cast<const uint32_t*>(out.contents);
  for (unsigned y = 0; y < height; ++y)
    for (unsigned x = 0; x < width; ++x) {
      unsigned i = y * width + x, lx = x % 8, ly = y % 4;
      const uint32_t want[8] = {x, y, lx, ly, ly * 8 + lx, x / 8, y / 4, 100 * gx + gy};
      for (unsigned k = 0; k < 8; ++k) check(v[i * 8 + k] == want[k], test, "id", i * 8 + k, v[i * 8 + k], want[k]);
    }
  return true;
}

bool testTables(const Context& c, const char* path) {
  const char* test = "tables";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 64);
  if (!pso) return false;
  const unsigned n = 4 * 64;
  std::vector<uint32_t> init(n, 0xFFFFFFFFu);
  id<MTLBuffer> out = makeBuffer(c, init.data(), n * 4);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:out offset:0 atIndex:0];
  [e dispatchThreadgroups:MTLSizeMake(4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t table[8] = {11, 13, 17, 19, 23, 29, 31, 37}, prv[4] = {1, 3, 5, 7}, scratch[4] = {5, 6, 7, 8};
  const uint32_t* v = static_cast<const uint32_t*>(out.contents);
  for (unsigned g = 0; g < n; ++g) {
    uint32_t want = table[g % 8] + 100 * prv[(g / 8) % 4] + 10000 * scratch[(g + 1) % 4];
    check(v[g] == want, test, "lookup", g, v[g], want);
  }
  return true;
}

bool testShared(const Context& c, const char* path) {
  const char* test = "shared";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}, {0, 2, 2, 0}};
  options.pushConstantBuffer = 3;
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 64);
  if (!pso) return false;
  const unsigned groups = 5, n = groups * 64;
  std::vector<uint32_t> in(n), outInit(n, 0), sumsInit(groups, 0);
  for (unsigned i = 0; i < n; ++i) in[i] = i * 7 + 3;
  id<MTLBuffer> bin = makeBuffer(c, in.data(), n * 4), bout = makeBuffer(c, outInit.data(), n * 4),
                bsums = makeBuffer(c, sumsInit.data(), groups * 4);
  const uint32_t add = 1000;
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e setBuffer:bsums offset:0 atIndex:2];
  [e setBytes:&add length:sizeof(add) atIndex:3];
  [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* out = static_cast<const uint32_t*>(bout.contents);
  const uint32_t* sums = static_cast<const uint32_t*>(bsums.contents);
  for (unsigned g = 0; g < groups; ++g) {
    uint32_t sum = 0;
    for (unsigned l = 0; l < 64; ++l) {
      uint32_t want = in[g * 64 + 63 - l] + add;
      check(out[g * 64 + l] == want, test, "reversed", g * 64 + l, out[g * 64 + l], want);
      sum += in[g * 64 + l];
    }
    check(sums[g] == sum, test, "sum", g, sums[g], sum);
  }
  return true;
}

bool testAtomics(const Context& c, const char* path) {
  const char* test = "atomics";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 32);
  if (!pso) return false;
  const unsigned groups = 3;
  uint32_t counters[9] = {0, 0, 0, 0, 0, 0xFFFFFFFFu, 0, 0, 0};
  uint32_t outInit[groups] = {};
  id<MTLBuffer> bc = makeBuffer(c, counters, sizeof(counters)), bout = makeBuffer(c, outInit, sizeof(outInit));
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bc offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* v = static_cast<const uint32_t*>(bc.contents);
  const uint32_t want[9] = {24, 24, 24, 24, 31, 0, 0xFFFFFFFFu, 32, uint32_t(-int(groups * 32))};
  for (unsigned i = 0; i < 9; ++i) check(v[i] == want[i], test, "counter", i, v[i], want[i]);
  const uint32_t* o = static_cast<const uint32_t*>(bout.contents);
  for (unsigned g = 0; g < groups; ++g) check(o[g] == 32, test, "shared counter", g, o[g], 32);
  return true;
}

bool testBits(const Context& c, const char* path) {
  const char* test = "bits";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 64);
  if (!pso) return false;
  const unsigned n = 64;
  std::vector<uint32_t> in(n);
  const uint32_t edge[] = {0,           1,           2,           3,           0x80000000u, 0x7FFFFFFFu, 0xFFFFFFFFu,
                           0xFFFFFFFEu, 0x80000001u, 0x00010000u, 0xC0000000u, 0x40000000u, 0xFFFF0000u, 0x0000FFFFu};
  for (unsigned i = 0; i < n; ++i) in[i] = i < sizeof(edge) / 4 ? edge[i] : (0x9E3779B9u * (i + 1)) >> (i % 29);
  std::vector<uint32_t> outInit(n * 4, 0xDEADBEEFu);
  id<MTLBuffer> bin = makeBuffer(c, in.data(), n * 4), bout = makeBuffer(c, outInit.data(), n * 16);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* o = static_cast<const uint32_t*>(bout.contents);
  auto umsb = [](uint32_t v) { return v ? uint32_t(31 - __builtin_clz(v)) : 0xFFFFFFFFu; };
  for (unsigned i = 0; i < n; ++i) {
    uint32_t x = in[i], s = uint32_t(int32_t(x) < 0 ? ~x : x);
    check(o[4 * i] == umsb(x), test, "FindUMsb", i, o[4 * i], umsb(x));
    check(o[4 * i + 1] == umsb(s), test, "FindSMsb", i, o[4 * i + 1], umsb(s));
    uint32_t lsb = x ? uint32_t(__builtin_ctz(x)) : 0xFFFFFFFFu;
    check(o[4 * i + 2] == lsb, test, "FindILsb", i, o[4 * i + 2], lsb);
    check(o[4 * i + 3] == umsb(x >> 1), test, "FindUMsb vector", i, o[4 * i + 3], umsb(x >> 1));
  }
  return true;
}

// Check both packing policies against the host IEEE conversion. The RTZ model
// starts with the host's nearest-even half and steps down only if it overshot.
// Exercise every finite half interval on both signs, its midpoint neighbors,
// and independent full-width float patterns.
bool testPack(const Context& c, const char* path, spirv2air::HalfPackingRounding rounding) {
  bool rtz = rounding == spirv2air::HalfPackingRounding::TowardZero;
  const char* test = rtz ? "pack-rtz" : "pack-rne";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  options.halfPackingRounding = rounding;
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 64);
  if (!pso) return false;
  const float inf = __builtin_inff(), nan = __builtin_nanf("");
  std::vector<float> in = {0, -0.0f, 65504, 65519.996f, 65520, -65520, 70000, 131008, -131008,
                          inf, -inf, nan, 3.0e-8f, -3.0e-8f, 1e-10f, -1e-10f, 1e30f, -1e30f};
  auto fromHalf = [](uint16_t bits) { _Float16 h; memcpy(&h, &bits, 2); return float(h); };
  for (uint32_t sign : {0u, 0x8000u}) {
    for (uint32_t magnitude = 0; magnitude < 0x7BFFu; ++magnitude) {
      float low = fromHalf(uint16_t(sign | magnitude));
      float high = fromHalf(uint16_t(sign | (magnitude + 1)));
      float midpoint = (low + high) * 0.5f;
      in.insert(in.end(), {low, std::nextafter(midpoint, low), midpoint, std::nextafter(midpoint, high)});
    }
  }
  uint32_t random = 1;
  for (unsigned i = 0; i < (1u << 18); ++i) {
    random = random * 1664525u + 1013904223u;
    float value; memcpy(&value, &random, 4); in.push_back(value);
  }
  while (in.size() % 128) in.push_back(0);
  const unsigned n = unsigned(in.size() / 2);
  std::vector<uint32_t> outInit(n, 0xDEADBEEFu);
  id<MTLBuffer> bin = makeBuffer(c, in.data(), in.size() * 4), bout = makeBuffer(c, outInit.data(), n * 4);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(n / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* o = static_cast<const uint32_t*>(bout.contents);
  auto half = [&](float f) {
    _Float16 h = static_cast<_Float16>(f);
    uint16_t pattern; memcpy(&pattern, &h, 2);
    if (rtz && std::isfinite(f) && std::fabs(float(h)) > std::fabs(f)) --pattern;
    return uint32_t(pattern);
  };
  auto isNan = [](uint32_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x3FF) != 0; };
  for (unsigned g = 0; g < n; ++g) {
    for (unsigned lane = 0; lane < 2; ++lane) {
      uint32_t want = half(in[2 * g + lane]), got = (o[g] >> (16 * lane)) & 0xFFFF;
      check(got == want || (isNan(got) && isNan(want)), test, lane ? "high half" : "low half", g, got, want);
    }
  }
  printf("%s: %zu conversions checked\n", test, in.size());
  return true;
}

// A NoContraction OpFMul in a compute shader: the correctly rounded product with a -0 result turned into +0 (the
// fma(x, y, +0) form of a precise product), compared bit for bit with the CPU's fmaf(x, y, 0). Inputs have full-width
// mantissas so a fused or double-rounded product would differ, and avoid subnormal results (compute modules flush
// them).
bool testStrictMul(const Context& c, const char* path, bool explicitNone = false) {
  const char* test = explicitNone ? "strictmul-none" : "strictmul";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 64);
  if (!pso) return false;
  const float inf = __builtin_inff(), nan = __builtin_nanf("");
  const unsigned n = 64;
  std::vector<float> in(2 * n);
  const float special[][2] = {{-0.0f, 1.0f}, {1.0f, -0.0f},  {-1.5f, 0.0f},     {-0.0f, -0.0f},
                              {0.0f, 0.0f},  {inf, 2.0f},    {inf, -2.0f},      {inf, 0.0f},
                              {nan, 1.0f},   {1e20f, 1e20f}, {1e-20f, -1e-10f}, {3.0f, 5.0f}};
  for (unsigned g = 0; g < n; ++g) {
    if (g < sizeof(special) / sizeof(special[0])) {
      in[2 * g] = special[g][0];
      in[2 * g + 1] = special[g][1];
    } else {
      uint32_t a = 0x3F800001u + g * 0x2491u, b = 0x3F7FFFFFu - g * 0x1357u;  // 1 + k ulp, with full mantissas
      memcpy(&in[2 * g], &a, 4);
      memcpy(&in[2 * g + 1], &b, 4);
      if (g & 1) in[2 * g] = -in[2 * g];
    }
  }
  std::vector<uint32_t> outInit(n, 0xDEADBEEFu);
  id<MTLBuffer> bin = makeBuffer(c, in.data(), in.size() * 4), bout = makeBuffer(c, outInit.data(), n * 4);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* o = static_cast<const uint32_t*>(bout.contents);
  for (unsigned g = 0; g < n; ++g) {
    volatile float left = in[2 * g], right = in[2 * g + 1];
    float want = explicitNone ? left * right : __builtin_fmaf(left, right, 0.0f), got;
    memcpy(&got, &o[g], 4);
    uint32_t wantBits;
    memcpy(&wantBits, &want, 4);
    check(o[g] == wantBits || (got != got && want != want), test, "NoContraction product", g, o[g], wantBits);
  }
  return true;
}

uint32_t bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

// IEEE and wrapping-integer SPIR-V semantics (tests/data/compute/ieee.spvasm).
bool testIeee(const Context& c, const char* path) {
  const char* test = "ieee";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 64);
  if (!pso) return false;
  const unsigned n = 64;
  auto f = [](uint32_t u) {
    float v;
    memcpy(&v, &u, 4);
    return v;
  };
  const uint32_t xs[] = {0xAF47E430u, 0x80800001u, 0x7FC00000u, 0x3F800000u, 0x7FC00000u, 0x00000000u,
                         0x80000000u, 0x3F800000u, 0x40000000u, 0xBF000000u, 0xFF800000u, 0x7F800000u,
                         0x3E800000u, 0xB3800000u, 0x80000000u, 0xC0000000u};
  const uint32_t ys[] = {0x3F800000u, 0x3F800000u, 0x3F800000u, 0x7FC00000u, 0x7FC00000u, 0x80000000u,
                         0x00000000u, 0x3F800000u, 0x3F800000u, 0xBF000000u, 0x7F800000u, 0x7F800000u,
                         0x80000000u, 0x00000000u, 0x80000000u, 0xC0000000u};
  const uint32_t ps[] = {0x80000000u, 1,           2,           3, 0xFFFFFFFFu, 0x7FFFFFFFu,
                         0x12345678u, 0xC0B0A0F1u, 0x80000001u, 0, 0xE0000000u, 5};
  std::vector<uint32_t> in(n * 4);
  for (unsigned i = 0; i < n; ++i) {
    in[4 * i] = xs[i % 16];
    in[4 * i + 1] = ys[(i + i / 16) % 16];
    in[4 * i + 2] = ps[i % 12];
    in[4 * i + 3] = i * 7;
  }
  std::vector<uint32_t> outInit(n * 8, 0xDEADBEEFu);
  id<MTLBuffer> bin = makeBuffer(c, in.data(), n * 16), bout = makeBuffer(c, outInit.data(), n * 32);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* o = static_cast<const uint32_t*>(bout.contents);
  for (unsigned i = 0; i < n; ++i) {
    float x = f(in[4 * i]), y = f(in[4 * i + 1]);
    uint32_t p = in[4 * i + 2];
    bool nan = x != x || y != y;
    float fr = x - floorf(x);
    uint32_t wantFract = bits(fr);
    bool fractOk =
        o[8 * i] == wantFract || (fr != fr && (o[8 * i] & 0x7F800000u) == 0x7F800000u && (o[8 * i] & 0x7FFFFFu));
    check(fractOk, test, "Fract (x - floor(x))", i, o[8 * i], wantFract);
    check(o[8 * i + 1] == (!nan && x != y), test, "FOrdNotEqual", i, o[8 * i + 1], !nan && x != y);
    check(o[8 * i + 2] == (nan || x == y), test, "FUnordEqual", i, o[8 * i + 2], nan || x == y);
    uint32_t wantMul = 0u - (0x80000000u * p);
    check(o[8 * i + 3] == wantMul, test, "-(INT_MIN * p) wraps", i, o[8 * i + 3], wantMul);
    bool wantGt = int32_t(uint32_t(-1065166664) - p) > 8388608;
    check(o[8 * i + 4] == wantGt, test, "ISub wraps", i, o[8 * i + 4], wantGt);
    check(o[8 * i + 5] == 0, test, "BitFieldSExtract Count 0", i, o[8 * i + 5], 0);
    // FMin/FMax of +0 and -0 are left open (the spec's ordering does not distinguish them), as are NaN operands.
    if (!nan && !(x == 0 && y == 0)) {
      uint32_t wantMin = bits(y < x ? y : x), wantMax = bits(y > x ? y : x);
      check(o[8 * i + 6] == wantMin, test, "FMin (y < x ? y : x)", i, o[8 * i + 6], wantMin);
      check(o[8 * i + 7] == wantMax, test, "FMax (y > x ? y : x)", i, o[8 * i + 7], wantMax);
    }
  }
  return true;
}

float fromBits(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

uint16_t halfBits(float f) {
  _Float16 h = (_Float16)f;
  uint16_t u;
  memcpy(&u, &h, 2);
  return u;
}

float halfFloat(uint16_t u) {
  _Float16 h;
  memcpy(&h, &u, 2);
  return float(h);
}

bool testCoverage(const Context& c, const char* path, unsigned family) {
  const char* test = family == 0 ? "integer" : family == 1 ? "float_math" : "packing";
  const unsigned n = 64, stride = family == 0 ? 49 : family == 1 ? 36 : 31;
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, n);
  if (!pso) return false;
  const uint32_t integers[][4] = {{0, 0, 0, 0},
                                  {0x80000000, 0xFFFFFFFF, 0, 1},
                                  {0xFFFFFFFF, 0x80000000, 0xFFFFFFFF, 2},
                                  {7, uint32_t(-3), 3, 0xFFFFFFFF},
                                  {uint32_t(-7), 3, uint32_t(-3), 1},
                                  {0x7FFFFFFF, 1, 32, 32},
                                  {0xFFFFFFFF, 33, 63, 0}};
  const float floats[][4] = {{7.5f, -2.0f, 0.25f, -0.5f},
                             {-7.5f, 2.0f, -0.25f, 0.5f},
                             {0.0f, -0.0f, 0.0f, 1.0f},
                             {1.0f, 1.0f, -1.0f, 1.0f},
                             {INFINITY, -INFINITY, 1.0f, 0.0f},
                             {NAN, 0.0f, -1.0f, 0.0f},
                             {0x1p-149f, -0x1p-126f, 0x1p-24f, 0.0f},
                             {65520.0f, -65520.0f, 2.0f, 1.0f},
                             {1.000244140625f, 1.00048828125f, -1.0f, 0.0f}};
  const float packs[][4] = {{-1, 0, 1, 0.5f},
                            {-2, 2, -0.5f, 0.25f},
                            {-0.0f, 0.0f, 0.125f, 0.875f},
                            {0.1f, -0.1f, 0.3f, -0.3f},
                            {0x1p-24f, -0x1p-24f, 0x1p-149f, -0x1p-149f},
                            {65504, -65504, INFINITY, -INFINITY},
                            {NAN, INFINITY, -INFINITY, -0.0f}};
  std::vector<uint32_t> input(n * 4), initial(n * stride, 0xDEADBEEF);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned k = 0; k < 4; ++k) {
      if (family == 0)
        input[i * 4 + k] = i < 7 ? integers[i][k] : (0x9E3779B9u * (i + k + 1));
      else
        input[i * 4 + k] = bits(family == 1 ? floats[i % 9][k] : packs[i % 7][k]);
    }
  id<MTLBuffer> bin = makeBuffer(c, input.data(), input.size() * 4);
  id<MTLBuffer> bout = makeBuffer(c, initial.data(), initial.size() * 4);
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(n, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* output = static_cast<const uint32_t*>(bout.contents);
  for (unsigned row = 0; row < n; ++row) {
    const uint32_t* u = input.data() + row * 4;
    unsigned col = 0;
    auto integer = [&](uint32_t want) {
      uint32_t got = output[row * stride + col++];
      check(got == want, test, "integer", row * stride + col - 1, got, want);
    };
    auto equivalent = [](uint32_t got, float want, bool approximate) {
      float g = fromBits(got);
      if (std::isnan(want)) return std::isnan(g);
      if (got == bits(want)) return true;
      return approximate && std::isfinite(want) && std::isfinite(g) &&
             std::fabs(g - want) <= 4e-6f * std::max(std::fabs(want), 0x1p-126f);
    };
    auto floating = [&](float want, bool approximate = false) {
      uint32_t got = output[row * stride + col++];
      check(equivalent(got, want, approximate), test, "float", row * stride + col - 1, got, bits(want));
    };
    auto floatingEither = [&](float want, float flushed, bool approximate = false) {
      // AIR advertises denorms_disable; the Shader profile does not request DenormPreserve.
      if (equivalent(output[row * stride + col], flushed, approximate))
        ++col;
      else
        floating(want, approximate);
    };
    auto ftz = [](float v) { return std::fabs(v) < 0x1p-126f ? std::copysign(0.0f, v) : v; };
    if (family == 0) {
      for (unsigned op = 0; op < 5; ++op)
        for (unsigned k = 0; k < 2; ++k) {
          int64_t a = int32_t(u[k]), b = int32_t(u[k + 1]);
          int64_t r = b ? a % b : a;
          if (op == 0) integer(b ? uint32_t(a / b) : 0xFFFFFFFFu);
          if (op == 1) integer(uint32_t(r));
          if (op == 2) integer(uint32_t(r && ((r < 0) != (b < 0)) ? r + b : r));
          if (op == 3) integer(u[k + 1] ? u[k] / u[k + 1] : 0xFFFFFFFFu);
          if (op == 4) integer(u[k + 1] ? u[k] % u[k + 1] : u[k]);
        }
      for (unsigned op = 0; op < 4; ++op)
        for (unsigned hi = 0; hi < 2; ++hi)
          for (unsigned k = 0; k < 2; ++k) {
            uint64_t a = u[k], b = u[k + 1], v;
            if (op == 0)
              v = a + b;
            else if (op == 1)
              v = hi ? uint64_t(a < b) << 32 : a - b;
            else if (op == 2)
              v = a * b;
            else
              v = uint64_t(int64_t(int32_t(a)) * int64_t(int32_t(b)));
            integer(uint32_t(hi ? v >> 32 : v));
          }
      for (unsigned k = 0; k < 2; ++k) {
        uint32_t reverse = 0;
        for (unsigned bit = 0; bit < 32; ++bit) reverse |= ((u[k] >> bit) & 1u) << (31 - bit);
        integer(reverse);
      }
      for (unsigned k = 0; k < 2; ++k) integer(__builtin_popcount(u[k]));
      integer(u[3] < 2 ? u[u[3]] : 0);
      for (unsigned k = 0; k < 2; ++k) integer(u[3] == k ? u[2] : u[k]);
      for (unsigned k = 0; k < 2; ++k) integer(int32_t(u[k]) < int32_t(u[k + 1]) ? u[k] : u[k + 1]);
      for (unsigned k = 0; k < 2; ++k) integer(int32_t(u[k]) < 0 ? 0u - u[k] : u[k]);
      for (unsigned k = 0; k < 2; ++k) integer(int32_t(u[k]) < 0 ? 0xFFFFFFFFu : uint32_t(u[k] != 0));
      for (unsigned k = 0; k < 2; ++k) integer(std::min(std::max(u[k], u[k + 1]), u[k + 2]));
      for (unsigned op = 0; op < 3; ++op)
        for (unsigned k = 0; k < 2; ++k) {
          unsigned shift = u[k + 1] & 31;
          integer(op == 0 ? u[k] << shift : op == 1 ? u[k] >> shift : uint32_t(int32_t(u[k]) >> shift));
        }
      integer(u[0]);
      integer(u[1]);
    } else if (family == 1) {
      float f[4];
      for (unsigned k = 0; k < 4; ++k) f[k] = fromBits(u[k]);
      for (unsigned k = 0; k < 2; ++k)
        floatingEither(std::fmod(f[k], f[k + 1]), ftz(std::fmod(ftz(f[k]), ftz(f[k + 1]))));
      for (unsigned k = 0; k < 2; ++k) {
        volatile float product = f[k + 1] * std::floor(f[k] / f[k + 1]);
        float a = ftz(f[k]), b = ftz(f[k + 1]);
        volatile float flushedProduct = ftz(b * std::floor(ftz(a / b)));
        floatingEither(f[k] - product, ftz(a - flushedProduct));
      }
      for (unsigned k = 0; k < 2; ++k) floating(f[k] < 0 ? -1 : f[k] > 0 ? 1 : 0);
      for (unsigned k = 0; k < 2; ++k) floating(std::exp(f[k]), true);
      for (unsigned k = 0; k < 2; ++k) floatingEither(std::log(f[k]), std::log(ftz(f[k])), true);
      for (unsigned k = 0; k < 2; ++k) {
        // Pow leaves negative bases, 0^nonpositive and NaN operands undefined in GLSL.std.450.
        if (f[k] < 0 || (f[k] == 0 && f[k + 1] <= 0) || std::isnan(f[k]) || std::isnan(f[k + 1]) ||
            std::fpclassify(f[k]) == FP_SUBNORMAL || std::fpclassify(f[k + 1]) == FP_SUBNORMAL)
          ++col;
        else
          floating(std::pow(f[k], f[k + 1]), true);
      }
      for (unsigned k = 0; k < 2; ++k)
        floatingEither(std::fma(f[k], f[k + 1], f[k + 2]), ftz(std::fma(ftz(f[k]), ftz(f[k + 1]), ftz(f[k + 2]))));
      for (unsigned k = 0; k < 2; ++k) {
        float h = halfFloat(halfBits(f[k]));
        floating(std::fabs(h) < 0x1p-14f ? std::copysign(0.0f, h) : h);
      }
      for (unsigned k = 0; k < 2; ++k) integer(std::isinf(f[k]) ? u[k + 2] : u[k]);
      auto modf = [&](bool whole) {
        for (unsigned k = 0; k < 2; ++k) {
          float integral;
          float fraction = std::modf(f[k], &integral);
          floating(whole ? integral : fraction);
        }
      };
      auto frexp = [&](bool exponent) {
        for (unsigned k = 0; k < 2; ++k) {
          int exp = 0;
          float frac = std::isfinite(f[k]) ? std::frexp(f[k], &exp) : f[k];
          if (exponent)
            integer(uint32_t(exp));
          else
            floating(frac);
        }
      };
      modf(false);
      modf(true);
      frexp(false);
      frexp(true);
      modf(false);
      modf(true);
      frexp(false);
      frexp(true);
      for (unsigned k = 0; k < 2; ++k)
        floatingEither(std::ldexp(f[k], k ? 2 : -1), ftz(std::ldexp(ftz(f[k]), k ? 2 : -1)));
    } else {
      float f[4];
      for (unsigned k = 0; k < 4; ++k) f[k] = fromBits(u[k]);
      for (unsigned kind = 0; kind < 4; ++kind) {
        bool sign = !(kind & 1);
        unsigned width = kind < 2 ? 8 : 16, count = 32 / width;
        uint32_t mask = (1u << width) - 1, maximum = sign ? mask >> 1 : mask, packed = 0;
        for (unsigned k = 0; k < count; ++k) {
          float clamped = std::isnan(f[k]) ? 0 : std::min(std::max(f[k], sign ? -1.0f : 0.0f), 1.0f);
          // The product is rounded once, from its exact value: a float multiply first would turn 0.3f * 255 (76.500003)
          // into the tie 76.5 and then round it to even.
          int32_t v = int32_t(std::nearbyint(double(clamped) * double(maximum)));
          packed |= (uint32_t(v) & mask) << (width * k);
        }
        integer(packed);
        for (uint32_t raw : {packed, u[3]})
          for (unsigned k = 0; k < count; ++k) {
            uint32_t field = (raw >> (width * k)) & mask;
            int32_t v = sign && (field & (1u << (width - 1))) ? int32_t(field) - int32_t(mask + 1) : int32_t(field);
            floating(std::max(float(v) / float(maximum), sign ? -1.0f : 0.0f), true);
          }
      }
      uint32_t packed = 0;
      // PackHalf2x16 is the IEEE binary16 conversion: round to nearest even, overflow to Inf, NaN stays NaN.
      for (unsigned k = 0; k < 2; ++k) packed |= uint32_t(halfBits(f[k])) << (k * 16);
      integer(packed);
      floating(halfFloat(uint16_t(packed)));
      floating(halfFloat(uint16_t(packed >> 16)));
    }
    check(col == stride, test, "reference width", row, col, stride);
  }
  return true;
}

bool testFetch(const Context& c, const char* path) {
  const char* test = "fetch";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 1}, {1, 2, 0, 2}};
  options.pushConstantBuffer = 1;
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 16);
  if (!pso) return false;

  // Float texture: 4x4 with a second level of 2x2.
  MTLTextureDescriptor* fd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                width:4
                                                                               height:4
                                                                            mipmapped:YES];
  fd.mipmapLevelCount = 2;
  id<MTLTexture> tf = [c.device newTextureWithDescriptor:fd];
  float level0[4 * 4 * 4], level1[2 * 2 * 4];
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
      float t[4] = {float(x + 10 * y), 0.5f, -float(x), 1.0f};
      memcpy(&level0[(y * 4 + x) * 4], t, 16);
    }
  for (int y = 0; y < 2; ++y)
    for (int x = 0; x < 2; ++x) {
      float t[4] = {float(100 + x + 10 * y), -2.0f, 3.5f * float(y), 8.0f};
      memcpy(&level1[(y * 2 + x) * 4], t, 16);
    }
  [tf replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:level0 bytesPerRow:64];
  [tf replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 withBytes:level1 bytesPerRow:32];

  MTLTextureDescriptor* ud = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Uint
                                                                                width:4
                                                                               height:4
                                                                            mipmapped:NO];
  id<MTLTexture> tu = [c.device newTextureWithDescriptor:ud];
  uint32_t texelsU[4 * 4 * 4];
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
      uint32_t t[4] = {uint32_t(x * 3 + y * 5), 0xDEADBEEFu, uint32_t(y), uint32_t(x ^ y)};
      memcpy(&texelsU[(y * 4 + x) * 4], t, 16);
    }
  [tu replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texelsU bytesPerRow:64];

  // Multisampled texture: cleared to one colour by a render pass with no draws.
  MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                width:4
                                                                               height:4
                                                                            mipmapped:NO];
  md.textureType = MTLTextureType2DMultisample;
  md.sampleCount = 4;
  md.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
  md.storageMode = MTLStorageModePrivate;
  id<MTLTexture> tm = [c.device newTextureWithDescriptor:md];
  if (!tm) return fprintf(stderr, "%s: no 4x multisampled RGBA16Float texture\n", test), false;

  const unsigned count = 16 * 6 * 4;
  std::vector<uint32_t> init(count, 0xFFFFFFFFu);
  id<MTLBuffer> out = makeBuffer(c, init.data(), count * 4);

  for (uint32_t sample = 0; sample < 4; ++sample) {
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    if (sample == 0) {
      MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
      rp.colorAttachments[0].texture = tm;
      rp.colorAttachments[0].loadAction = MTLLoadActionClear;
      rp.colorAttachments[0].storeAction = MTLStoreActionStore;
      rp.colorAttachments[0].clearColor = MTLClearColorMake(0.25, 0.5, 0.75, 1.0);
      [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    }
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:out offset:0 atIndex:0];
    [e setBytes:&sample length:sizeof(sample) atIndex:1];
    [e setTexture:tf atIndex:0];
    [e setTexture:tu atIndex:1];
    [e setTexture:tm atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
    [e endEncoding];
    if (!wait(cb, test)) return false;
    const uint32_t* v = static_cast<const uint32_t*>(out.contents);
    for (unsigned y = 0; y < 4; ++y)
      for (unsigned x = 0; x < 4; ++x) {
        unsigned i = (y * 4 + x) * 24;
        uint32_t want[24] = {};
        for (unsigned k = 0; k < 4; ++k) {
          want[k] = bits(level0[(y * 4 + x) * 4 + k]);
          want[4 + k] = texelsU[(y * 4 + x) * 4 + k];
          want[12 + k] = bits(level1[((y & 1) * 2 + (x & 1)) * 4 + k]);
        }
        const float ms[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        for (unsigned k = 0; k < 4; ++k) want[8 + k] = bits(ms[k]);
        want[16] = 2;
        want[17] = 2;
        want[18] = 2;
        want[19] = 4;
        want[20] = 4;
        want[21] = 4;
        for (unsigned k = 0; k < 24; ++k) {
          char what[32];
          snprintf(what, sizeof(what), "sample %u texel", sample);
          check(v[i + k] == want[k], test, what, i + k, v[i + k], want[k]);
        }
      }
  }
  return true;
}

bool testAtomicExtra(const Context& c, const char* path) {
  const char* test = "atomic_extra";
  spirv2air::Options options;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  id<MTLComputePipelineState> pso = makePipeline(c, test, path, options, 1);
  if (!pso) return false;
  uint32_t initial[18] = {};
  id<MTLBuffer> counter = makeBuffer(c, initial, 4), out = makeBuffer(c, initial, sizeof(initial));
  id<MTLCommandBuffer> cb = [c.queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:counter offset:0 atIndex:0];
  [e setBuffer:out offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  [e endEncoding];
  if (!wait(cb, test)) return false;
  const uint32_t* o = static_cast<const uint32_t*>(out.contents);
  for (unsigned space = 0; space < 2; ++space) {
    // CPU state machine: store, load, increment, decrement, successful CAS, failed CAS, second failed CAS.
    uint32_t value = 0xFFFFFFFFu;
    uint32_t want[9];
    want[0] = value;
    want[1] = value++;
    want[2] = value--;
    want[3] = value;
    if (value == 0xFFFFFFFFu) value = 9;
    want[4] = value;
    want[5] = value;
    if (value == 7) value = 3;
    want[6] = value;
    want[7] = value;
    if (value == 3) value = 4;
    want[8] = value;
    for (unsigned k = 0; k < 9; ++k)
      check(o[space * 9 + k] == want[k], test, "old/new value", space * 9 + k, o[space * 9 + k], want[k]);
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 15)
    return fprintf(stderr, "usage: compute_gpu ids.spv shared.spv atomics.spv fetch.spv bits.spv ieee.spv pack.spv "
                           "strictmul.spv integer.spv float_math.spv packing.spv atomic_extra.spv strictmul_none.spv "
                           "tables.spv\n"),
           2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 2;
    Context c{device, [device newCommandQueue]};
    bool ran = testIds(c, argv[1]);
    ran = testShared(c, argv[2]) && ran;
    ran = testAtomics(c, argv[3]) && ran;
    ran = testFetch(c, argv[4]) && ran;
    ran = testBits(c, argv[5]) && ran;
    ran = testIeee(c, argv[6]) && ran;
    ran = testPack(c, argv[7], spirv2air::HalfPackingRounding::NearestEven) && ran;
    ran = testPack(c, argv[7], spirv2air::HalfPackingRounding::TowardZero) && ran;
    ran = testStrictMul(c, argv[8]) && ran;
    for (unsigned family = 0; family < 3; ++family) ran = testCoverage(c, argv[9 + family], family) && ran;
    ran = testAtomicExtra(c, argv[12]) && ran;
    ran = testStrictMul(c, argv[13], true) && ran;
    ran = testTables(c, argv[14]) && ran;
    if (!ran || failures)
      return fprintf(stderr, "compute_gpu: %d value failures%s\n", failures, ran ? "" : ", setup failed"), 1;
    printf("compute_gpu: ids, barriers, atomics, IEEE, numeric ops, packing, image queries and tables match\n");
  }
  return 0;
}
