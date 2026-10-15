// Checks spirv2air's lowering of gl_CullDistance on the GPU. Metal has clip distances only,
// so each cull distance becomes a clip distance that is -1 at the vertices where it is negative and 2^30 elsewhere.
// A primitive must disappear when, for some cull distance, all its vertices are negative, and otherwise must cover
// exactly the pixels it covers with no cull distances at all. Each random triangle is drawn twice into its own
// target, once by the translated vertex shader (Position, ClipDistance[1], CullDistance[2]) and once by an MSL
// reference vertex shader that writes the same position and clip distance and no cull distances.
// usage: cull_gpu <cull.vert.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace {

constexpr int kSize = 64;       // render target width and height
constexpr int kTriangles = 400;  // triangles per run
constexpr int kTlabBuffer = 2;

std::vector<uint32_t> readWords(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(f ? size_t(f.tellg()) / 4 : 0);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
  return words;
}

struct Data {
  float pos[64][4];
  float data[64][4];  // cull0, cull1, clip0, unused
};

NSString* const kSource = @R"(
#include <metal_stdlib>
using namespace metal;
struct Data { float4 pos[64]; float4 data[64]; };
struct VOut { float4 pos [[position]]; float clip [[clip_distance]] [1]; };
vertex VOut reference(uint v [[vertex_id]], constant Data& d [[buffer(3)]]) {
  VOut o;
  o.pos = d.pos[v];
  o.clip[0] = d.data[v].z;
  return o;
}
fragment float4 white() { return float4(1); }
)";

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 2) return fprintf(stderr, "usage: cull_gpu <cull.vert.spv>\n"), 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::vector<uint32_t> words = readWords(argv[1]);
    spirv2air::Result r = spirv2air::translate(words.data(), words.size());
    if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), 1;
    NSError* error = nil;
    dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> library = [device newLibraryWithData:blob error:&error];
    if (!library) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), 1;
    id<MTLLibrary> source = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!source) return fprintf(stderr, "MSL: %s\n", error.description.UTF8String), 1;

    auto pipeline = [&](id<MTLFunction> vertex) {
      MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = vertex;
      d.fragmentFunction = [source newFunctionWithName:@"white"];
      d.colorAttachments[0].pixelFormat = MTLPixelFormatR8Unorm;
      NSError* e = nil;
      id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:d error:&e];
      if (!p) fprintf(stderr, "pipeline: %s\n", e.description.UTF8String);
      return p;
    };
    id<MTLRenderPipelineState> translated = pipeline([library newFunctionWithName:@"main"]);
    id<MTLRenderPipelineState> reference = pipeline([source newFunctionWithName:@"reference"]);
    if (!translated || !reference) return 1;

    // Triangle t has vertices 3t..3t+2 of the 64-entry table: each draw uses a fresh table.
    std::mt19937 rng(12345);
    auto uniform = [&](float lo, float hi) { return lo + (hi - lo) * float(rng() & 0xFFFFFF) / float(0x1000000); };
    const float cullValues[] = {-1000.0f, -1.0f, -1e-3f, -1e-30f, 0.0f, 1e-30f, 1e-3f, 1.0f, 1000.0f};
    int culledExpected = 0, drawnExpected = 0, partial = 0;
    struct Case {
      Data data;
      bool culled;
    };
    std::vector<Case> cases(kTriangles);
    for (int t = 0; t < kTriangles; ++t) {
      Data& d = cases[t].data;
      memset(&d, 0, sizeof(d));
      float cx = uniform(-1.0f, 1.0f), cy = uniform(-1.0f, 1.0f);
      float scale = (t % 4 == 0) ? 0.05f : (t % 4 == 1) ? 0.3f : (t % 4 == 2) ? 1.0f : 3.0f;
      bool culled = false;
      int neg[2] = {0, 0};
      for (int v = 0; v < 3; ++v) {
        d.pos[v][0] = cx + uniform(-scale, scale);
        d.pos[v][1] = cy + uniform(-scale, scale);
        d.pos[v][2] = uniform(0.0f, 1.0f);
        d.pos[v][3] = 1.0f;
        // Clip distance: positive, except in a quarter of the cases where a plane at x = -0.5 clips the triangle.
        d.data[v][2] = (t % 4 == 3) ? d.pos[v][0] + 0.5f : 1.0f;
        for (int k = 0; k < 2; ++k) {
          // Bias the draw towards all-negative so that every case is common.
          bool bias = (t % 3 == k) || (t % 5 == 4);
          float value = bias && uniform(0, 1) < 0.8f ? cullValues[rng() % 4] : cullValues[rng() % 9];
          d.data[v][k] = value;
          neg[k] += value < 0;
        }
      }
      culled = neg[0] == 3 || neg[1] == 3;
      cases[t].culled = culled;
      (culled ? culledExpected : drawnExpected)++;
      partial += !culled && (neg[0] || neg[1]);
    }

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                   width:kSize
                                                                                  height:kSize
                                                                               mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];
    int mismatches = 0, culledDrawn = 0, wrongPixels = 0, sliverFree = 0;
    for (int t = 0; t < kTriangles; ++t) {
      id<MTLBuffer> table = [device newBufferWithBytes:&cases[t].data
                                                length:sizeof(Data)
                                               options:MTLResourceStorageModeShared];
      uint64_t address = table.gpuAddress;
      memcpy(tlab.contents, &address, 8);  // uniform binding 0
      id<MTLTexture> targets[2];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      for (int which = 0; which < 2; ++which) {
        targets[which] = [device newTextureWithDescriptor:td];
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.colorAttachments[0].texture = targets[which];
        pass.colorAttachments[0].loadAction = MTLLoadActionClear;
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
        pass.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
        [enc setRenderPipelineState:which ? reference : translated];
        [enc setVertexBuffer:tlab offset:0 atIndex:kTlabBuffer];
        [enc setVertexBuffer:table offset:0 atIndex:3];
        [enc useResource:table usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        [enc endEncoding];
      }
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), 1;
      std::vector<uint8_t> got(kSize * kSize), want(kSize * kSize);
      [targets[0] getBytes:got.data() bytesPerRow:kSize fromRegion:MTLRegionMake2D(0, 0, kSize, kSize) mipmapLevel:0];
      [targets[1] getBytes:want.data() bytesPerRow:kSize fromRegion:MTLRegionMake2D(0, 0, kSize, kSize) mipmapLevel:0];
      int diff = 0, drawn = 0, wantDrawn = 0;
      for (int i = 0; i < kSize * kSize; ++i) {
        drawn += got[i] != 0;
        wantDrawn += want[i] != 0;
        diff += (got[i] != 0) != (want[i] != 0);
      }
      if (cases[t].culled) {
        culledDrawn += drawn != 0;
        if (drawn) wrongPixels += drawn;
        if (drawn && ++mismatches <= 10)
          fprintf(stderr, "FAIL triangle %d: culled but %d pixels drawn\n", t, drawn);
      } else {
        sliverFree += diff == 0;
        wrongPixels += diff;
        if (diff && ++mismatches <= 10)
          fprintf(stderr, "FAIL triangle %d: %d pixels differ from the unculled draw (%d vs %d)\n", t, diff, drawn,
                  wantDrawn);
      }
    }
    printf("cull_gpu: %d triangles culled, %d drawn (%d of them with some negative vertex); %d culled drawn, %d identical, "
           "%d wrong pixels\n",
           culledExpected, drawnExpected, partial, culledDrawn, sliverFree, wrongPixels);
    return mismatches ? 1 : 0;
  }
}
