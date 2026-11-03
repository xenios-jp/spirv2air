// Centroid from the primitive's raster coverage (tests/data/frag/rastercentroid.frag.spvasm: barycentrics and their
// derivatives under a SIMD-group vote, the input sample mask, FindILsb and InterpolateAtSample) on 2X and 4X targets.
// - Random triangles: the translated shader writes the same bits as the same algorithm in MSL and as MSL's hardware
//   centroid, which with nothing hidden is the first covered sample, or the pixel center at full coverage.
// - A triangle drawn before a nearer one in the same pass keeps, on its surviving samples, the translated values it
//   gets alone. Apple's hidden-surface removal can take hidden samples out of the input mask before shading; the test
//   reports how many of MSL's hardware-centroid values move under the nearer triangle.
#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <spirv2air.h>
#include <vector>

namespace {

constexpr int kTlab = 2;
constexpr unsigned kSize = 32, kTargets = 2;

NSString* const kSource = @R"(
  #include <metal_stdlib>
  using namespace metal;
  struct V {
    float4 position [[position]];
    float4 a [[user(texcoord0)]];
    float4 b [[user(texcoord1)]];
  };
  // v[i] = (x, y in pixels, depth, w); both attributes are (x, y, 2x - y, 1).
  vertex V vtx(uint id [[vertex_id]], constant float4* v [[buffer(1)]]) {
    float4 p = v[id];
    V o;
    o.position = float4(p.x / 16 - 1, 1 - p.y / 16, p.z, 1) * p.w;
    o.a = o.b = float4(p.x, p.y, 2 * p.x - p.y, 1);
    return o;
  }
  struct Params {
    float4 sp[4];
    uint4 n;
  };
  struct Out {
    float4 o0 [[color(0)]];
    float4 o1 [[color(1)]];
    uint mask [[sample_mask]];
  };
  struct In {
    float3 bary [[barycentric_coord, center_no_perspective]];
    interpolant<float4, interpolation::perspective> a [[user(texcoord0)]];
    interpolant<float4, interpolation::no_perspective> b [[user(texcoord1)]];
  };
  fragment Out same_algorithm(In in [[stage_in]], uint mask_in [[sample_mask]], constant Params& p [[buffer(3)]]) {
    Out o;
    o.o0 = in.a.interpolate_at_center();
    o.o1 = in.b.interpolate_at_center();
    if (simd_any(mask_in != p.n.y)) {
      float3 dx = dfdx(in.bary), dy = dfdy(in.bary);
      uint covered = mask_in;
      for (uint i = 0; i < p.n.x; ++i)
        if (all(in.bary + dx * (p.sp[i].x - 0.5) + dy * (p.sp[i].y - 0.5) >= 0)) covered |= 1u << i;
      if (covered != p.n.y) {
        o.o0 = in.a.interpolate_at_sample(ctz(covered));
        o.o1 = in.b.interpolate_at_sample(ctz(covered));
      }
    }
    o.mask = 0xffffffff;
    return o;
  }
  struct Centroid {
    float4 a [[user(texcoord0), centroid_perspective]];
    float4 b [[user(texcoord1), centroid_no_perspective]];
  };
  fragment Out hardware_centroid(Centroid in [[stage_in]]) { return {in.a, in.b, 0xffffffff}; }
  fragment Out occluder() { return {float4(-1), float4(-1), 0xffffffff}; }
  kernel void readback(texture2d_ms<float, access::read> t [[texture(0)]], device float4* out [[buffer(0)]],
                       constant uint& samples [[buffer(1)]], uint3 id [[thread_position_in_grid]]) {
    out[(id.y * 32 + id.x) * samples + id.z] = t.read(id.xy, id.z);
  }
)";

struct Params {
  float sp[4][4];
  uint32_t n[4];
};

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !device.supportsPullModelInterpolation || !device.supportsShaderBarycentricCoordinates) return 77;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
    if (words.size() < 5) return 2;
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
    auto result = spirv2air::translate(words.data(), words.size());
    if (!result.ok()) return std::fprintf(stderr, "%s\n", result.error.c_str()), 1;
    NSError* error = nil;
    dispatch_data_t blob = dispatch_data_create(result.metallib.data(), result.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> translated = [device newLibraryWithData:blob error:&error];
    if (!translated) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
    id<MTLLibrary> reference = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!reference) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLComputePipelineState> reader =
        [device newComputePipelineStateWithFunction:[reference newFunctionWithName:@"readback"] error:&error];
    if (!reader) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
    MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
    dd.depthCompareFunction = MTLCompareFunctionLess;
    dd.depthWriteEnabled = YES;
    id<MTLDepthStencilState> depth = [device newDepthStencilStateWithDescriptor:dd];
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> coordinate(-4 * 256, 36 * 256), wbits(128, 512);
    size_t compared = 0, partial = 0, hidden = 0, shifted = 0;
    for (unsigned samples : {2u, 4u}) {
      std::vector<MTLSamplePosition> positions(samples);
      [device getDefaultSamplePositions:positions.data() count:samples];
      Params params = {};
      for (unsigned i = 0; i < samples; ++i) params.sp[i][0] = positions[i].x, params.sp[i][1] = positions[i].y;
      params.n[0] = samples;
      params.n[1] = (1u << samples) - 1;
      id<MTLBuffer> ubo = [device newBufferWithBytes:&params
                                              length:sizeof(params)
                                             options:MTLResourceStorageModeShared];
      id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];
      uint64_t address = ubo.gpuAddress;
      std::memset(tlab.contents, 0, 112);
      std::memcpy(tlab.contents, &address, 8);
      // Arms: 0 the same algorithm in MSL, 1 translated, 2 MSL's hardware centroid; 3 the occluder.
      id<MTLRenderPipelineState> pipelines[4];
      NSString* names[4] = {@"same_algorithm", nil, @"hardware_centroid", @"occluder"};
      for (unsigned arm = 0; arm < 4; ++arm) {
        MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [reference newFunctionWithName:@"vtx"];
        descriptor.fragmentFunction =
            arm == 1 ? [translated newFunctionWithName:@"main"] : [reference newFunctionWithName:names[arm]];
        descriptor.rasterSampleCount = samples;
        for (unsigned t = 0; t < kTargets; ++t) descriptor.colorAttachments[t].pixelFormat = MTLPixelFormatRGBA32Float;
        descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        pipelines[arm] = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!pipelines[arm]) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
      }
      // Draws the triangles (4 floats per vertex) with `arm`, then `occluded` with the occluder, in one pass.
      auto render = [&](unsigned arm, const std::vector<float>& triangle, const std::vector<float>& occluded) {
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                      width:kSize
                                                                                     height:kSize
                                                                                  mipmapped:NO];
        td.sampleCount = samples;
        td.textureType = MTLTextureType2DMultisample;
        td.storageMode = MTLStorageModePrivate;
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        id<MTLTexture> targets[kTargets];
        for (unsigned t = 0; t < kTargets; ++t) {
          targets[t] = [device newTextureWithDescriptor:td];
          pass.colorAttachments[t].texture = targets[t];
          pass.colorAttachments[t].loadAction = MTLLoadActionClear;
          pass.colorAttachments[t].clearColor = MTLClearColorMake(-7, -7, -7, -7);
          pass.colorAttachments[t].storeAction = MTLStoreActionStore;
        }
        td.pixelFormat = MTLPixelFormatDepth32Float;
        td.usage = MTLTextureUsageRenderTarget;
        pass.depthAttachment.texture = [device newTextureWithDescriptor:td];
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = 1;
        pass.depthAttachment.storeAction = MTLStoreActionDontCare;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
        [encoder setDepthStencilState:depth];
        [encoder setRenderPipelineState:pipelines[arm]];
        [encoder setVertexBytes:triangle.data() length:triangle.size() * 4 atIndex:1];
        [encoder setFragmentBuffer:tlab offset:0 atIndex:kTlab];
        [encoder useResource:ubo usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        [encoder setFragmentBytes:&params length:sizeof(params) atIndex:3];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:triangle.size() / 4];
        if (!occluded.empty()) {
          [encoder setRenderPipelineState:pipelines[3]];
          [encoder setVertexBytes:occluded.data() length:occluded.size() * 4 atIndex:1];
          [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:occluded.size() / 4];
        }
        [encoder endEncoding];
        NSUInteger bytes = kSize * kSize * samples * 16;
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes * kTargets options:MTLResourceStorageModeShared];
        id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
        [compute setComputePipelineState:reader];
        [compute setBytes:&samples length:4 atIndex:1];
        for (unsigned t = 0; t < kTargets; ++t) {
          [compute setTexture:targets[t] atIndex:0];
          [compute setBuffer:buffer offset:t * bytes atIndex:0];
          [compute dispatchThreads:MTLSizeMake(kSize, kSize, samples) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        }
        [compute endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t* p = static_cast<const uint32_t*>(buffer.contents);
        return cb.error ? std::vector<uint32_t>() : std::vector<uint32_t>(p, p + bytes * kTargets / 4);
      };
      // Random triangles, vertices on the 1/256 pixel grid, w from 0.5 to 2.
      for (unsigned k = 0; k < 40; ++k) {
        std::vector<float> triangle;
        for (unsigned v = 0; v < 3; ++v)
          triangle.insert(triangle.end(),
                          {coordinate(rng) / 256.0f, coordinate(rng) / 256.0f, 0.5f, wbits(rng) / 256.0f});
        std::vector<uint32_t> out[3];
        for (unsigned arm = 0; arm < 3; ++arm)
          if ((out[arm] = render(arm, triangle, {})).empty()) return std::fprintf(stderr, "render failed\n"), 1;
        for (unsigned arm : {0u, 2u})
          if (out[1] != out[arm]) {
            for (size_t i = 0; i < out[1].size(); ++i)
              if (out[1][i] != out[arm][i]) {
                size_t texel = i / 4, plane = kSize * kSize * samples, pixel = texel % plane / samples;
                std::fprintf(stderr,
                             "%uX triangle %u: %s differs at target %zu pixel (%zu, %zu) sample %zu: %08x, "
                             "translated %08x\n",
                             samples, k, arm ? "hardware centroid" : "same algorithm in MSL", texel / plane,
                             pixel % kSize, pixel / kSize, texel % samples, out[arm][i], out[1][i]);
                break;
              }
            return 1;
          }
        compared += out[1].size();
        // Partly covered pixels: some sample keeps the clear value.
        for (size_t pixel = 0; pixel < kSize * kSize; ++pixel) {
          unsigned written = 0;
          for (unsigned s = 0; s < samples; ++s) written += out[1][(pixel * samples + s) * 4 + 3] != 0xc0e00000;
          partial += written && written < samples;
        }
      }
      // A covers the target at depth 0.5; the nearer B (depth 0.25) cuts its pixels along three edges.
      std::vector<float> a = {-16, -16, 0.5f, 1, 80, -16, 0.5f, 1, -16, 80, 0.5f, 1};
      std::vector<float> b = {3.3125f, 2.0625f, 0.25f, 1, 29.6875f, 9.875f, 0.25f, 1, 12.375f, 30.1875f, 0.25f, 1};
      for (unsigned arm : {1u, 2u}) {
        std::vector<uint32_t> alone = render(arm, a, {}), occluded = render(arm, a, b);
        if (alone.empty() || occluded.empty()) return std::fprintf(stderr, "render failed\n"), 1;
        for (size_t i = 0; i < alone.size(); i += 4) {
          if (occluded[i] == 0xbf800000) {  // B's
            hidden += arm == 1;
            continue;
          }
          if (!std::memcmp(&alone[i], &occluded[i], 16)) continue;
          if (arm == 2) {
            ++shifted;
            continue;
          }
          size_t texel = i / 4, plane = kSize * kSize * samples, pixel = texel % plane / samples;
          std::fprintf(stderr,
                       "%uX: A's translated value at target %zu pixel (%zu, %zu) sample %zu changes under B: "
                       "%08x alone, %08x in B's pass\n",
                       samples, texel / plane, pixel % kSize, pixel / kSize, texel % samples, alone[i], occluded[i]);
          return 1;
        }
      }
    }
    if (!partial || !hidden) return std::fprintf(stderr, "no partly covered or hidden samples\n"), 1;
    // shifted counts the samples where MSL's hardware centroid moves under B (Apple's hidden-surface removal).
    std::printf("PASS: %zu words match MSL's same algorithm and hardware centroid at 2X and 4X (%zu partly covered "
                "pixels); under a nearer triangle %zu samples hidden, translated values unchanged where hardware "
                "centroid changes %zu\n",
                compared, partial, hidden, shifted);
    return 0;
  }
}
