// Checks a graphics StorageBuffer outside the shared-memory/counter slots (GraphicsBindings::storageBuffers), as
// xenia's tile cull bounds use it: tilecull.vert reads int pairs with plain loads and updates them with signed atomic
// min/max in a vertex function. The buffer's address is the pointer at byte 112 of the argument buffer (bound at buffer
// 2). usage: tilecull_gpu <tilecull.vert.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <climits>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {

NSString* const kSource = @R"(
#include <metal_stdlib>
using namespace metal;
fragment float4 frag() { return float4(1); }
)";

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 2) return fprintf(stderr, "usage: tilecull_gpu <tilecull.vert.spv>\n"), 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
    std::vector<uint32_t> words(f ? size_t(f.tellg()) / 4 : 0);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
    spirv2air::Options options;
    // Without the option the storage buffer is rejected.
    if (spirv2air::translate(words.data(), words.size(), options).ok()) return fprintf(stderr, "FAIL: accepted\n"), 1;
    options.graphics.storageBuffers.push_back({0, 4, 112});
    spirv2air::Result r = spirv2air::translate(words.data(), words.size(), options);
    if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), 1;
    NSError* error = nil;
    dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> lib = [device newLibraryWithData:blob error:&error];
    if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), 1;
    id<MTLLibrary> source = [device newLibraryWithSource:kSource options:nil error:&error];
    MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = [lib newFunctionWithName:@"main"];
    d.fragmentFunction = [source newFunctionWithName:@"frag"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
    id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:d error:&error];
    if (!pso) return fprintf(stderr, "pipeline: %s\n", error.description.UTF8String), 1;

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:4
                                                                                 height:4
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    id<MTLBuffer> bounds = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> tlab = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    int32_t* b = static_cast<int32_t*>(bounds.contents);
    for (int s = 0; s < 4; ++s) b[2 * s] = INT_MAX, b[2 * s + 1] = INT_MIN;
    *reinterpret_cast<uint64_t*>(static_cast<uint8_t*>(tlab.contents) + 112) = bounds.gpuAddress;

    MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture = target;
    pass.colorAttachments[0].loadAction = MTLLoadActionClear;
    pass.colorAttachments[0].storeAction = MTLStoreActionDontCare;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
    [enc setRenderPipelineState:pso];
    [enc setVertexBuffer:tlab offset:0 atIndex:2];
    [enc useResource:bounds usage:MTLResourceUsageRead | MTLResourceUsageWrite stages:MTLRenderStageVertex];
    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:48];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) return fprintf(stderr, "GPU: %s\n", cb.error.description.UTF8String), 1;

    // Vertices 0..47: slot s sees i = s, s + 4, ..., s + 44 with v = 7 i - 20.
    int failures = 0;
    for (int s = 0; s < 4; ++s) {
      int lo = 7 * s - 20, hi = 7 * (s + 44) - 20;
      if (b[2 * s] != lo || b[2 * s + 1] != hi) {
        fprintf(stderr, "FAIL slot %d: min %d max %d, expected %d %d\n", s, b[2 * s], b[2 * s + 1], lo, hi);
        ++failures;
      }
    }
    printf(failures ? "tilecull_gpu: %d FAILED\n" : "tilecull_gpu: all passed\n", failures);
    return failures ? 1 : 0;
  }
}
