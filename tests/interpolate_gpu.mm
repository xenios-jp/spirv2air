// Pull-model interpolation (GLSL.std.450 InterpolateAtOffset, InterpolateAtSample and InterpolateAtCentroid) against
// native MSL interpolants on 2X and 4X targets: the translated pixel shader must write the same bits as MSL's, and at
// fully covered pixels the values at the offsets, the sample positions and the centroid (the pixel center there).
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <spirv2air.h>
#include <vector>

namespace {

constexpr int kTlab = 2;
constexpr unsigned kSize = 16, kTargets = 4;

NSString* const kSource = @R"(
  #include <metal_stdlib>
  using namespace metal;
  struct V {
    float4 position [[position]];
    float4 a [[user(texcoord0)]];
    float4 b [[user(texcoord1)]];
  };
  // The triangle (0, 0), (16, 0), (0, 16) in pixels: its hypotenuse x + y = 16 cuts the pixels with x + y = 15.
  // Both attributes are (x, y, 2x - y, 1) in pixels; w[i] scales vertex i's clip coordinates.
  vertex V vtx(uint id [[vertex_id]], constant float4& w [[buffer(0)]]) {
    const float2 xy[3] = {float2(0, 0), float2(16, 0), float2(0, 16)};
    float2 p = xy[id];
    V v;
    v.position = float4(p.x / 8 - 1, 1 - p.y / 8, 0, 1) * w[id];
    v.a = v.b = float4(p.x, p.y, 2 * p.x - p.y, 1);
    return v;
  }
  struct In {
    interpolant<float4, interpolation::perspective> a [[user(texcoord0)]];
    interpolant<float4, interpolation::no_perspective> b [[user(texcoord1)]];
  };
  struct Params {
    float2 offset;
    int sample;
  };
  struct Out {
    float4 c0 [[color(0)]];
    float4 c1 [[color(1)]];
    float4 c2 [[color(2)]];
    float4 c3 [[color(3)]];
  };
  // MSL's offsets are from the pixel's upper-left corner, SPIR-V's from its center.
  fragment Out reference(In in [[stage_in]], constant Params& p [[buffer(3)]]) {
    Out o;
    o.c0 = in.a.interpolate_at_offset(p.offset + 0.5);
    o.c1 = in.b.interpolate_at_sample(p.sample);
    o.c2 = in.b.interpolate_at_centroid();
    o.c3 = float4(in.a.interpolate_at_offset(p.offset + 0.5).x, in.b.interpolate_at_sample(p.sample).y,
                  in.a.interpolate_at_centroid().z, in.b.interpolate_at_center().w);
    return o;
  }
  kernel void readback(texture2d_ms<float, access::read> t [[texture(0)]], device float4* out [[buffer(0)]],
                       constant uint& samples [[buffer(1)]], uint3 id [[thread_position_in_grid]]) {
    out[(id.y * 16 + id.x) * samples + id.z] = t.read(id.xy, id.z);
  }
)";

struct Params {
  float offset[2];
  int sample;
  int pad;
};

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !device.supportsPullModelInterpolation) return 77;
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
    // Offsets on the 1/16 grid that InterpolateAtOffset guarantees, from the extremes of its range.
    const float offsets[][2] = {{0, 0}, {-0.5f, -0.5f}, {0.4375f, -0.25f}, {0.25f, 0.125f}, {-0.125f, 0.4375f}};
    // w = 1 everywhere makes perspective interpolation linear in pixels; the other draw is perspective.
    const float ws[][4] = {{1, 1, 1, 1}, {0.5f, 2, 1, 1}};
    size_t compared = 0, checked = 0, partial = 0;
    float worst = 0;
    for (unsigned samples : {2u, 4u}) {
      std::vector<MTLSamplePosition> positions(samples);
      [device getDefaultSamplePositions:positions.data() count:samples];
      id<MTLRenderPipelineState> pipelines[2];
      for (unsigned arm = 0; arm < 2; ++arm) {
        MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [reference newFunctionWithName:@"vtx"];
        descriptor.fragmentFunction =
            arm ? [translated newFunctionWithName:@"main"] : [reference newFunctionWithName:@"reference"];
        descriptor.rasterSampleCount = samples;
        for (unsigned t = 0; t < kTargets; ++t) descriptor.colorAttachments[t].pixelFormat = MTLPixelFormatRGBA32Float;
        pipelines[arm] = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!pipelines[arm]) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
      }
      for (unsigned draw = 0; draw < 2; ++draw) {
        for (unsigned k = 0; k < sizeof(offsets) / sizeof(offsets[0]); ++k) {
          Params params = {{offsets[k][0], offsets[k][1]}, int(k % samples), 0};
          id<MTLBuffer> ubo = [device newBufferWithBytes:&params
                                                  length:sizeof(params)
                                                 options:MTLResourceStorageModeShared];
          id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];
          uint64_t address = ubo.gpuAddress;
          std::memset(tlab.contents, 0, 112);
          std::memcpy(tlab.contents, &address, 8);
          std::vector<uint32_t> outputs[2];
          for (unsigned arm = 0; arm < 2; ++arm) {
            MTLTextureDescriptor* td =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
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
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
            [encoder setRenderPipelineState:pipelines[arm]];
            [encoder setVertexBytes:ws[draw] length:16 atIndex:0];
            if (arm) {
              [encoder setFragmentBuffer:tlab offset:0 atIndex:kTlab];
              [encoder useResource:ubo usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
            } else {
              [encoder setFragmentBytes:&params length:sizeof(params) atIndex:3];
            }
            [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
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
            if (cb.error) return std::fprintf(stderr, "%s\n", cb.error.description.UTF8String), 1;
            const uint32_t* p = static_cast<const uint32_t*>(buffer.contents);
            outputs[arm].assign(p, p + bytes * kTargets / 4);
          }
          if (outputs[0] != outputs[1]) {
            for (size_t i = 0; i < outputs[0].size(); ++i)
              if (outputs[0][i] != outputs[1][i]) {
                size_t texel = i / 4, plane = kSize * kSize * samples;
                std::fprintf(stderr,
                             "%uX draw %u case %u: target %zu pixel (%zu, %zu) sample %zu component %zu: MSL %08x, "
                             "translated %08x\n",
                             samples, draw, k, texel / plane, texel % plane / samples % kSize,
                             texel % plane / samples / kSize, texel % samples, i % 4, outputs[0][i], outputs[1][i]);
                break;
              }
            return 1;
          }
          compared += outputs[0].size();
          if (draw) continue;
          // Pixels with x + y <= 13 are fully covered: the values are the pixel coordinates of the position read.
          auto at = [&](unsigned t, unsigned x, unsigned y, unsigned c) {
            float f;
            std::memcpy(&f, &outputs[1][(((t * kSize + y) * kSize + x) * samples) * 4 + c], 4);
            return f;
          };
          const MTLSamplePosition& s = positions[params.sample];
          for (unsigned y = 0; y < kSize; ++y)
            for (unsigned x = 0; x < kSize; ++x) {
              if (x + y == 15 && (at(2, x, y, 0) != x + 0.5f || at(2, x, y, 1) != y + 0.5f)) ++partial;
              if (x + y > 13) continue;
              const float expected[kTargets][2] = {{x + 0.5f + offsets[k][0], y + 0.5f + offsets[k][1]},
                                                   {x + s.x, y + s.y},
                                                   {x + 0.5f, y + 0.5f},
                                                   {x + 0.5f + offsets[k][0], y + s.y}};
              for (unsigned t = 0; t < kTargets; ++t) {
                float want[4] = {expected[t][0], expected[t][1], 2 * expected[t][0] - expected[t][1], 1};
                if (t == 3) want[2] = 2 * (x + 0.5f) - (y + 0.5f);
                for (unsigned c = 0; c < 4; ++c) {
                  float e = std::fabs(at(t, x, y, c) - want[c]);
                  worst = std::fmax(worst, e);
                  // A 1/64 pixel bound: well inside the 1/16 grid, so it pins the offset, its axes and its sign.
                  if (!(e <= 1.0f / 64)) {
                    std::fprintf(stderr, "%uX case %u: target %u pixel (%u, %u) component %u: %.9g, expected %.9g\n",
                                 samples, k, t, x, y, c, at(t, x, y, c), want[c]);
                    return 1;
                  }
                  ++checked;
                }
              }
            }
        }
      }
    }
    // The hypotenuse's pixels are partly covered at 2X and 4X, so InterpolateAtCentroid moves off their center.
    if (!partial) return std::fprintf(stderr, "no centroid away from a pixel center\n"), 1;
    std::printf("PASS: %zu words match MSL at 2X and 4X; %zu values at known positions within %.3g pixels; %zu "
                "partly covered centroids\n",
                compared, checked, worst, partial);
    return 0;
  }
}
