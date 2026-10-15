// Draws one tessellated patch with spirv2air's post-tessellation vertex functions and
// checks what reaches the fragment shader: the control-point values of the right patch, TessCoord's third
// component, PrimitiveId, and the vertex positions (through which pixels are covered).
// usage: tess_gpu <quad.tese.spv> <triangle.tese.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

constexpr int kSize = 16;  // render target width and height
constexpr int kPatch = 1;  // the patch drawn (patch 0 is skipped, so patch_id must include patchStart)
constexpr int kControlPointsBuffer = 6;
constexpr int kTlabBuffer = 2;

std::vector<uint32_t> readWords(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(f ? size_t(f.tellg()) / 4 : 0);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
  return words;
}

int failures = 0;
void check(bool ok, const char* what, int x, int y, float got, float want) {
  if (ok) return;
  if (++failures <= 10) fprintf(stderr, "FAIL %s at (%d, %d): %g, expected %g\n", what, x, y, got, want);
}

// Runs one domain. Control point k of patch p holds 10 * p + k.
bool run(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLFunction> fragment, const char* path, bool quad) {
  const char* name = quad ? "quad" : "triangle";
  const unsigned n = quad ? 4 : 3;
  std::vector<uint32_t> words = readWords(path);
  spirv2air::Result r = spirv2air::translate(words.data(), words.size());
  if (!r.ok()) return fprintf(stderr, "%s: translate failed: %s\n", name, r.error.c_str()), false;

  NSError* error = nil;
  dispatch_data_t data = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  if (!library) return fprintf(stderr, "%s: newLibraryWithData: %s\n", name, error.description.UTF8String), false;
  id<MTLFunction> vertex = [library newFunctionWithName:@"main"];
  if (!vertex) return fprintf(stderr, "%s: no function main\n", name), false;
  if (vertex.patchType != (quad ? MTLPatchTypeQuad : MTLPatchTypeTriangle) ||
      vertex.patchControlPointCount != NSInteger(n))
    return fprintf(stderr, "%s: patch type %d with %ld control points\n", name, int(vertex.patchType),
                   long(vertex.patchControlPointCount)),
           false;

  MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
  desc.vertexFunction = vertex;
  desc.fragmentFunction = fragment;
  desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
  desc.maxTessellationFactor = 64;
  desc.tessellationPartitionMode = MTLTessellationPartitionModeInteger;
  desc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
  desc.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
  id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:desc error:&error];
  if (!pipeline) return fprintf(stderr, "%s: pipeline: %s\n", name, error.description.UTF8String), false;

  std::vector<float> points;
  for (int p = 0; p <= kPatch; ++p)
    for (unsigned k = 0; k < n; ++k) points.push_back(float(10 * p + k));
  id<MTLBuffer> controlPoints = [device newBufferWithBytes:points.data()
                                                    length:points.size() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
  // Factor 1 everywhere: the patch is drawn as its corners only, so every value is interpolated linearly.
  std::vector<uint16_t> factors((kPatch + 1) * (quad ? 6 : 4), 0x3C00);  // half 1.0
  id<MTLBuffer> factorBuffer = [device newBufferWithBytes:factors.data()
                                                   length:factors.size() * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
  std::vector<uint8_t> tlabBytes(112, 0);
  id<MTLBuffer> tlab = [device newBufferWithBytes:tlabBytes.data()
                                           length:tlabBytes.size()
                                          options:MTLResourceStorageModeShared];

  MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                width:kSize
                                                                               height:kSize
                                                                            mipmapped:NO];
  td.usage = MTLTextureUsageRenderTarget;
  td.storageMode = MTLStorageModeShared;
  id<MTLTexture> target = [device newTextureWithDescriptor:td];
  MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
  pass.colorAttachments[0].texture = target;
  pass.colorAttachments[0].loadAction = MTLLoadActionClear;
  pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
  pass.colorAttachments[0].storeAction = MTLStoreActionStore;

  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
  [enc setRenderPipelineState:pipeline];
  [enc setVertexBuffer:tlab offset:0 atIndex:kTlabBuffer];
  [enc setVertexBuffer:controlPoints offset:0 atIndex:kControlPointsBuffer];
  [enc setTessellationFactorBuffer:factorBuffer offset:0 instanceStride:0];
  [enc drawPatches:n
                  patchStart:kPatch
                  patchCount:1
            patchIndexBuffer:nil
      patchIndexBufferOffset:0
               instanceCount:1
                baseInstance:0];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.error) return fprintf(stderr, "%s: command buffer: %s\n", name, cb.error.description.UTF8String), false;

  std::vector<float> pixels(kSize * kSize * 4);
  [target getBytes:pixels.data() bytesPerRow:kSize * 16 fromRegion:MTLRegionMake2D(0, 0, kSize, kSize) mipmapLevel:0];
  int before = failures, covered = 0;
  for (int y = 0; y < kSize; ++y)
    for (int x = 0; x < kSize; ++x) {
      const float* px = &pixels[(y * kSize + x) * 4];
      // Pixel centre in TessCoord terms: x = 2u - 1, and NDC y points up while rows go down.
      float u = (x + 0.5f) / kSize, v = 1.0f - (y + 0.5f) / kSize;
      bool inside = quad || u + v < 1.0f;
      bool drawn = px[3] != -1.0f;
      if (std::fabs(u + v - 1.0f) < 1.5f / kSize && !quad) continue;  // pixels on the diagonal edge
      check(drawn == inside, "coverage", x, y, drawn, inside);
      if (!drawn || !inside) continue;
      ++covered;
      // Interpolating a constant need not be bit-exact, hence the tolerances.
      check(std::fabs(px[0] - float(10 * kPatch)) < 1e-3f, "control point 0", x, y, px[0], float(10 * kPatch));
      check(std::fabs(px[1] - float(10 * kPatch + n - 1)) < 1e-3f, "last control point", x, y, px[1],
            float(10 * kPatch + n - 1));
      float w = quad ? 0.0f : 1.0f - u - v;
      check(std::fabs(px[2] - w) < 1e-3f, "TessCoord.z", x, y, px[2], w);
      check(std::fabs(px[3] - float(kPatch)) < 1e-3f, "PrimitiveId", x, y, px[3], float(kPatch));
    }
  printf("%s: %d pixels checked, %s\n", name, covered, failures == before ? "ok" : "FAILED");
  return failures == before && covered > 0;
}

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 3) return fprintf(stderr, "usage: tess_gpu <quad.tese.spv> <triangle.tese.spv>\n"), 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    if (![device supportsFamily:MTLGPUFamilyMetal3]) return printf("tessellation not supported; skipped\n"), 0;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> fs = [device newLibraryWithSource:@"#include <metal_stdlib>\n"
                                                      "struct In { float4 v [[user(texcoord0)]]; };\n"
                                                      "fragment float4 f(In i [[stage_in]]) { return i.v; }\n"
                                             options:nil
                                               error:&error];
    if (!fs) return fprintf(stderr, "fragment shader: %s\n", error.description.UTF8String), 1;
    id<MTLFunction> fragment = [fs newFunctionWithName:@"f"];
    bool ok = run(device, queue, fragment, argv[1], true);
    ok &= run(device, queue, fragment, argv[2], false);
    return ok ? 0 : 1;
  }
}
