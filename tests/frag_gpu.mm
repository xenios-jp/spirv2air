// Checks spirv2air's sample-rate pixel shader inputs on the GPU, on a 4x MSAA target:
// gl_SampleID (air.sample_id), gl_SampleMaskIn (air.sample_mask_in), a Sample-qualified interpolator (air.sample),
// OpBitCount, OpIsHelperInvocationEXT and OpAtomicIAdd on the ZPD counter at buffer(7). The translated sample.frag is
// compared sample by sample against an MSL reference function, the counter against the reference's counter; demote.frag
// checks the SPIR-V semantics of the helper invocation query (true after a demote) against the known pixel count.
// fma.frag checks contraction: undecorated a*b-c becomes llvm.fmuladd as MSL's own contraction does (compared with
// the MSL reference bit for bit), NoContraction math stays strict (exactly 0).
// usage: frag_gpu <sample.frag.spv> <demote.frag.spv> <fma.frag.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr int kSize = 16;     // render target width and height
constexpr int kSamples = 4;
constexpr int kTlabBuffer = 2;
constexpr int kCounterBuffer = 7;

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
struct VOut { float4 pos [[position]]; float4 a [[user(texcoord0)]]; };
// Triangle 0 covers part of the target (partial pixels along its edges); triangle 1 covers all of it.
vertex VOut vtx(uint v [[vertex_id]], constant uint& tri [[buffer(0)]]) {
  const float2 small[3] = {float2(-0.8, -0.7), float2(0.9, -0.55), float2(-0.1, 0.9)};
  const float2 big[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  const float av[3] = {0.0, 1.0, 5.0};
  VOut o;
  o.pos = float4(tri ? big[v] : small[v], 0.5, 1);
  o.a = float4(av[v], 0, 0, 1);
  return o;
}
struct FIn { float4 pos [[position]]; float4 a [[user(texcoord0), sample_perspective]]; };
fragment float4 ref_sample(FIn in [[stage_in]], uint sid [[sample_id]], uint mask [[sample_mask]],
                           device atomic_uint* counter [[buffer(7)]]) {
  uint bc = popcount(mask * 1015u + sid * 65537u);
  if (!simd_is_helper_thread()) atomic_fetch_add_explicit(counter, popcount(mask), memory_order_relaxed);
  return float4(float(int(sid)), float(mask), float(bc), in.a.x);
}
struct FInC { float4 pos [[position]]; float4 a [[user(texcoord0)]]; };
fragment float4 ref_fma(FInC in [[stage_in]]) {
  float x = as_type<float>(as_type<uint>(in.a.x) | 0x2A5u);  // full-width mantissa, so x*x is inexact
  float p = x * x;
  float e = x * x - p;  // contracted to one llvm.fmuladd within the expression
  return float4(e, 0.0f, e + e, p);
}
kernel void resolve(texture2d_ms<float, access::read> t [[texture(0)]], device float4* out [[buffer(0)]],
                    uint3 gid [[thread_position_in_grid]]) {
  out[(gid.y * 16 + gid.x) * 4 + gid.z] = t.read(gid.xy, gid.z);
}
)";

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 4) return fprintf(stderr, "usage: frag_gpu <sample.frag.spv> <demote.frag.spv> <fma.frag.spv>\n"), 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    if (![device supportsTextureSampleCount:kSamples]) return fprintf(stderr, "no 4x MSAA\n"), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    MTLCompileOptions* co = [MTLCompileOptions new];
  #pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  co.fastMathEnabled = NO;  // the reference contracts only inside one expression, as the translator does
#pragma clang diagnostic pop
    id<MTLLibrary> source = [device newLibraryWithSource:kSource options:co error:&error];
    if (!source) return fprintf(stderr, "MSL: %s\n", error.description.UTF8String), 1;
    auto load = [&](const char* path) -> id<MTLFunction> {
      std::vector<uint32_t> words = readWords(path);
      spirv2air::Result r = spirv2air::translate(words.data(), words.size());
      if (!r.ok()) return fprintf(stderr, "translate %s failed: %s\n", path, r.error.c_str()), nil;
      dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
      NSError* e = nil;
      id<MTLLibrary> lib = [device newLibraryWithData:blob error:&e];
      if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", e.description.UTF8String), nil;
      return [lib newFunctionWithName:@"main"];
    };
    id<MTLFunction> sampleFn = load(argv[1]), demoteFn = load(argv[2]), fmaFn = load(argv[3]);
    if (!sampleFn || !demoteFn || !fmaFn) return 1;

    auto pipeline = [&](id<MTLFunction> fragment) -> id<MTLRenderPipelineState> {
      MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = [source newFunctionWithName:@"vtx"];
      d.fragmentFunction = fragment;
      d.rasterSampleCount = kSamples;
      d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
      NSError* e = nil;
      id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:d error:&e];
      if (!p) fprintf(stderr, "pipeline: %s\n", e.description.UTF8String);
      return p;
    };
    id<MTLRenderPipelineState> translated = pipeline(sampleFn);
    id<MTLRenderPipelineState> reference = pipeline([source newFunctionWithName:@"ref_sample"]);
    id<MTLRenderPipelineState> demote = pipeline(demoteFn);
    id<MTLRenderPipelineState> fmaState = pipeline(fmaFn);
    id<MTLRenderPipelineState> fmaRef = pipeline([source newFunctionWithName:@"ref_fma"]);
    if (!translated || !reference || !demote || !fmaState || !fmaRef) return 1;
    id<MTLComputePipelineState> resolve =
        [device newComputePipelineStateWithFunction:[source newFunctionWithName:@"resolve"] error:&error];

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                   width:kSize
                                                                                  height:kSize
                                                                               mipmapped:NO];
    td.textureType = MTLTextureType2DMultisample;
    td.sampleCount = kSamples;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModePrivate;
    id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];

    struct Run {
      std::vector<float> texels;  // [pixel][sample][4]
      uint32_t counter = 0;
    };
    // Draws one triangle (0: partial coverage, 1: whole target) with `state` and reads back every sample.
    auto draw = [&](id<MTLRenderPipelineState> state, uint32_t tri, Run& run) {
      id<MTLTexture> target = [device newTextureWithDescriptor:td];
      id<MTLBuffer> counter = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
      memset(counter.contents, 0, 4);
      id<MTLBuffer> out = [device newBufferWithLength:kSize * kSize * kSamples * 16 options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = target;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:state];
      [enc setVertexBytes:&tri length:4 atIndex:0];
      [enc setFragmentBuffer:tlab offset:0 atIndex:kTlabBuffer];
      [enc setFragmentBuffer:counter offset:0 atIndex:kCounterBuffer];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
      id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
      [ce setComputePipelineState:resolve];
      [ce setTexture:target atIndex:0];
      [ce setBuffer:out offset:0 atIndex:0];
      [ce dispatchThreads:MTLSizeMake(kSize, kSize, kSamples) threadsPerThreadgroup:MTLSizeMake(4, 4, kSamples)];
      [ce endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), false;
      run.texels.assign(static_cast<float*>(out.contents), static_cast<float*>(out.contents) + kSize * kSize * kSamples * 4);
      memcpy(&run.counter, counter.contents, 4);
      return true;
    };

    int failures = 0;
    for (uint32_t tri = 0; tri < 2; ++tri) {
      Run got, want;
      if (!draw(translated, tri, got) || !draw(reference, tri, want)) return 1;
      int diff = 0, covered = 0, sampleIds = 0;
      for (size_t i = 0; i < got.texels.size(); i += 4) {
        bool same = memcmp(&got.texels[i], &want.texels[i], 16) == 0;
        if (!same && ++diff <= 5)
          fprintf(stderr, "FAIL tri %u texel %zu: got (%g %g %g %g) want (%g %g %g %g)\n", tri, i / 4, got.texels[i],
                  got.texels[i + 1], got.texels[i + 2], got.texels[i + 3], want.texels[i], want.texels[i + 1],
                  want.texels[i + 2], want.texels[i + 3]);
        if (want.texels[i] >= 0) {
          ++covered;
          sampleIds += got.texels[i] == float((i / 4) % kSamples);
        }
      }
      // Per-sample shading: one sample per invocation, so SampleID is the sample's index (the mask is compared with the
      // reference above).
      if (sampleIds != covered) {
        fprintf(stderr, "FAIL tri %u: %d covered samples, %d with the sample id of their slot\n", tri, covered, sampleIds);
        ++failures;
      }
      if (got.counter != want.counter || !got.counter) {
        fprintf(stderr, "FAIL tri %u: counter %u, reference %u\n", tri, got.counter, want.counter);
        ++failures;
      }
      if (diff) ++failures;
      printf("frag_gpu: triangle %u: %d covered samples, %d differ from the reference, counter %u (reference %u)\n", tri,
             covered, diff, got.counter, want.counter);
    }

    // demote.frag over the whole target: sample 1 of each pixel is demoted and is then a helper invocation, every
    // other sample is not, so the counter is the number of pixels and sample 1 keeps the clear value.
    Run d;
    if (!draw(demote, 1, d)) return 1;
    int wrongSamples = 0;
    for (size_t i = 0; i < d.texels.size(); i += 4) wrongSamples += (d.texels[i] == 1.0f) != ((i / 4) % kSamples != 1);
    printf("frag_gpu: demote: counter %u (want %d), %d wrong samples\n", d.counter, kSize * kSize, wrongSamples);
    if (d.counter != kSize * kSize || wrongSamples) ++failures;

    // fma.frag over the whole target: bit-identical to the MSL reference (scalar and vector contraction), and the
    // NoContraction pair gives exactly 0.
    Run g, w;
    if (!draw(fmaState, 1, g) || !draw(fmaRef, 1, w)) return 1;
    int fmaDiff = 0, strictBad = 0, fusedNonzero = 0;
    for (size_t i = 0; i < g.texels.size(); i += 4) {
      fmaDiff += memcmp(&g.texels[i], &w.texels[i], 16) != 0 || g.texels[i + 1] != 0.0f;
      strictBad += g.texels[i + 1] != 0.0f;
      fusedNonzero += g.texels[i] != 0.0f && g.texels[i] != -1.0f;
    }
    printf("frag_gpu: fma: %d samples differ from the MSL reference, %d strict results not 0 (%d contracted results nonzero)\n",
           fmaDiff, strictBad, fusedNonzero);
    if (fmaDiff || strictBad) ++failures;
    return failures ? 1 : 0;
  }
}
