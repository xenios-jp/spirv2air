// Private memory forwarding in cseFunction, on the GPU: every case of tests/forward_cases.h is built twice, as is and
// after cseFunction with forwarding, compiled by Apple's compiler and run on every row of operands; both results must
// equal the host's value of the case's program, so a forward that is wrong for some indices, branch or loop count
// shows as a wrong value.
//
// Through the translator: forward.vert keeps a vec4 variable written one component at a time and a float[4] indexed
// with dynamic indices in memory, and stores and loads them in a branch and a loop; it is drawn once per row of
// (i, j, c, n) and its flat outputs are compared with the host's value.
// usage: forward_gpu <forward.vert.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cstdio>
#include <fstream>

#include "forward_cases.h"

namespace {

id<MTLComputePipelineState> pipeline(id<MTLDevice> dev, airkit::Kernel& k, std::string& err) {
  std::vector<uint8_t> lib = airkit::buildMetallib({{"k", k.bitcode()}});
  dispatch_data_t data = dispatch_data_create(lib.data(), lib.size(), nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  NSError* e = nil;
  id<MTLLibrary> l = [dev newLibraryWithData:data error:&e];
  if (!l) { err = e.localizedDescription.UTF8String; return nil; }
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:@"k"] error:&e];
  if (!p) err = e.localizedDescription.UTF8String;
  return p;
}

NSString* const kPass = @R"(
#include <metal_stdlib>
using namespace metal;
struct In {
  float4 r0 [[user(texcoord0), flat]];
  float4 r1 [[user(texcoord1), flat]];
};
struct Out {
  float4 r0 [[color(0)]];
  float4 r1 [[color(1)]];
};
fragment Out pass(In in [[stage_in]]) { return {in.r0, in.r1}; }
)";

// forward.vert's program on the host (small integers: every float operation is exact).
void vertexWant(uint32_t i, uint32_t j, uint32_t c, uint32_t n, float x, float y, float* want) {
  float v[4] = {x, y, x, x}, a[4] = {0, 0, 0, 0};
  a[i] = x;
  a[j] = y;
  if (c)
    v[2] = a[i];
  else
    v[3] = a[j] + 1;
  for (uint32_t k = 0; k < n; ++k) v[0] += 1, a[i] += v[1];
  for (int e = 0; e < 4; ++e) want[e] = v[e];
  want[4] = a[0], want[5] = a[1], want[6] = a[2] + a[3], want[7] = a[j];
}

int testVertex(id<MTLDevice> dev, id<MTLCommandQueue> q, const char* path) {
  @autoreleasepool {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
    spirv2air::Result r = spirv2air::translate(words.data(), words.size());
    if (!r.ok()) return std::printf("FAIL forward.vert: translate: %s\n", r.error.c_str()), 1;
    NSError* error = nil;
    dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> lib = [dev newLibraryWithData:blob error:&error];
    id<MTLLibrary> source = [dev newLibraryWithSource:kPass options:nil error:&error];
    if (!lib || !source) return std::printf("FAIL forward.vert: %s\n", error.description.UTF8String), 1;
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"main"];
    d.fragmentFunction = [source newFunctionWithName:@"pass"];
    d.colorAttachments[0].pixelFormat = d.colorAttachments[1].pixelFormat = MTLPixelFormatRGBA32Float;
    id<MTLRenderPipelineState> state = [dev newRenderPipelineStateWithDescriptor:d error:&error];
    if (!state) return std::printf("FAIL forward.vert: pipeline: %s\n", error.description.UTF8String), 1;
    // One pixel per row: draw r covers pixel (r, 0) and reads its uniform block through its own argument buffer.
    struct Uniforms {
      uint32_t i, j, c, n;
      float x, y, z, w;
    };
    std::vector<Uniforms> rows;
    for (uint32_t i = 0; i < 4; ++i)
      for (uint32_t j = 0; j < 4; ++j)
        for (uint32_t c = 0; c < 2; ++c)
          for (uint32_t n = 0; n < 3; ++n) {
            const float base = float(rows.size());
            rows.push_back({i, j, c, n, 100 + base, 7000 + base, 0, 0});
          }
    const NSUInteger count = rows.size();
    constexpr NSUInteger kTlabSize = 112;
    id<MTLBuffer> ubo = [dev newBufferWithBytes:rows.data() length:count * sizeof(Uniforms)
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> tlab = [dev newBufferWithLength:count * kTlabSize options:MTLResourceStorageModeShared];
    memset(tlab.contents, 0, count * kTlabSize);
    for (NSUInteger k = 0; k < count; ++k) {
      uint64_t address = ubo.gpuAddress + k * sizeof(Uniforms);
      memcpy(static_cast<uint8_t*>(tlab.contents) + k * kTlabSize, &address, 8);
    }
    id<MTLBuffer> heap = [dev newBufferWithLength:24 options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:count
                                                                                 height:1
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    id<MTLTexture> targets[2];
    for (int t = 0; t < 2; ++t) {
      targets[t] = [dev newTextureWithDescriptor:td];
      pass.colorAttachments[t].texture = targets[t];
      pass.colorAttachments[t].loadAction = MTLLoadActionClear;
      pass.colorAttachments[t].clearColor = MTLClearColorMake(-7, -7, -7, -7);
      pass.colorAttachments[t].storeAction = MTLStoreActionStore;
    }
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLRenderCommandEncoder> e = [cb renderCommandEncoderWithDescriptor:pass];
    [e setRenderPipelineState:state];
    [e setVertexBuffer:heap offset:0 atIndex:0];
    [e setVertexBuffer:heap offset:0 atIndex:1];
    for (NSUInteger k = 0; k < count; ++k) {
      [e setViewport:{double(k), 0, 1, 1, 0, 1}];
      [e setScissorRect:{k, 0, 1, 1}];
      [e setVertexBuffer:tlab offset:k * kTlabSize atIndex:2];
      [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    }
    [e useResource:ubo usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) return std::printf("FAIL forward.vert: %s\n", cb.error.description.UTF8String), 1;
    std::vector<float> texels[2] = {std::vector<float>(count * 4), std::vector<float>(count * 4)};
    for (int t = 0; t < 2; ++t)
      [targets[t] getBytes:texels[t].data() bytesPerRow:count * 16 fromRegion:MTLRegionMake2D(0, 0, count, 1)
               mipmapLevel:0];
    int bad = 0;
    for (NSUInteger k = 0; k < count; ++k) {
      const Uniforms& u = rows[k];
      float want[8];
      vertexWant(u.i, u.j, u.c, u.n, u.x, u.y, want);
      for (int o = 0; o < 8; ++o) {
        const float got = texels[o / 4][4 * k + o % 4];
        if (got != want[o] && bad++ < 4)
          std::printf("  forward.vert i=%u j=%u c=%u n=%u output %d: got %g, want %g\n", u.i, u.j, u.c, u.n, o, got,
                      want[o]);
      }
    }
    std::printf("%s forward.vert: %lu draws\n", bad ? "FAIL" : "ok  ", (unsigned long)count);
    return bad ? 1 : 0;
  }
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("no Metal device\n"); return 77; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (argc != 2) return std::printf("usage: forward_gpu <forward.vert.spv>\n"), 2;
    const std::vector<forwardcases::Row> rows = forwardcases::rows();
    const size_t n = rows.size();
    id<MTLBuffer> in = [dev newBufferWithBytes:rows.data() length:n * sizeof(forwardcases::Row)
                                       options:MTLResourceStorageModeShared];
    int failures = 0;
    for (const forwardcases::Case& c : forwardcases::cases()) {
      int bad = 0;
      for (int forward = 0; forward < 2; ++forward) {
        airkit::Kernel k("k");
        forwardcases::build(k, c);
        spirv2air::cseFunction(k.m, k.function(), forward);
        std::string err;
        id<MTLComputePipelineState> p = pipeline(dev, k, err);
        if (!p) {
          std::printf("FAIL %s: %s\n", c.name.c_str(), err.c_str());
          ++bad;
          continue;
        }
        id<MTLBuffer> out = [dev newBufferWithLength:n * 8 options:MTLResourceStorageModeShared];
        memset(out.contents, 0xEE, n * 8);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:p];
        [enc setBuffer:in offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        const uint32_t* got = static_cast<const uint32_t*>(out.contents);
        for (size_t r = 0; r < n; ++r) {
          const auto [w0, w1] = c.want(rows[r]);
          if ((got[2 * r] != w0 || got[2 * r + 1] != w1) && bad++ < 4) {
            const forwardcases::Row& o = rows[r];
            std::printf("  %s%s i=%u j=%u c=%u k=%u x=%u y=%u: got (%u, %u), want (%u, %u)\n", c.name.c_str(),
                        forward ? " (forwarded)" : "", o.i, o.j, o.c, o.k, o.x, o.y, got[2 * r], got[2 * r + 1], w0, w1);
          }
        }
      }
      std::printf("%s %s: %zu rows, as is and forwarded\n", bad ? "FAIL" : "ok  ", c.name.c_str(), n);
      failures += bad != 0;
    }
    failures += testVertex(dev, q, argv[1]);
    std::printf(failures ? "%d failing\n" : "all passed\n", failures);
    return failures ? 1 : 0;
  }
}
