// Pixel interlock on a raster-ordered buffer, against native MSL: 64 primitives
// covering the whole 16x16 target, in an attachment-less pass, each fold their
// index into every pixel's word non-atomically (word = word * 31 + index), so
// each word is right only when the fragments at its pixel ran in primitive
// order. MSL without raster_order_group is run for information only: whether
// it happens to come out ordered is up to the GPU.
#import <Metal/Metal.h>
#include <cstdio>
#include <fstream>
#include <spirv2air.h>
#include <vector>

constexpr unsigned kSize = 16, kPrimitives = 64, kRepeats = 8;

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || device.areRasterOrderGroupsSupported == NO) return 77;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
    if (words.size() < 5) return 2;
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
    NSString* source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      struct V {float4 position [[position]]; uint value [[user(texcoord0), flat]];};
      vertex V vertex_main(uint id [[vertex_id]]) {
        const float2 xy[3] = {float2(-1,-1),float2(3,-1),float2(-1,3)};
        return {float4(xy[id % 3],0,1), id / 3 + 1};
      }
      fragment void ordered(V v [[stage_in]], device uint* b [[buffer(8), raster_order_group(0)]]) {
        uint i = uint(v.position.y) * 16 + uint(v.position.x);
        b[i] = b[i] * 31 + v.value;
      }
      fragment void unordered(V v [[stage_in]], device uint* b [[buffer(8)]]) {
        uint i = uint(v.position.y) * 16 + uint(v.position.x);
        b[i] = b[i] * 31 + v.value;
      }
    )";
    NSError* error = nil;
    id<MTLLibrary> reference = [device newLibraryWithSource:source options:nil error:&error];
    if (!reference) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
    spirv2air::Options options;
    options.graphics.rasterOrderedBuffer = 8;
    auto result = spirv2air::translate(words.data(), words.size(), options);
    if (!result.ok()) return std::fprintf(stderr, "%s\n", result.error.c_str()), 1;
    dispatch_data_t blob = dispatch_data_create(result.metallib.data(), result.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> translated = [device newLibraryWithData:blob error:&error];
    if (!translated) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
    uint32_t expected = 0;
    for (uint32_t value = 1; value <= kPrimitives; ++value) expected = expected * 31 + value;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> top_level = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    unsigned unordered_wrong = 0;
    for (unsigned arm = 0; arm < 3; ++arm) {
      MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
      descriptor.vertexFunction = [reference newFunctionWithName:@"vertex_main"];
      descriptor.fragmentFunction = arm == 0   ? [translated newFunctionWithName:@"main"]
                                    : arm == 1 ? [reference newFunctionWithName:@"ordered"]
                                               : [reference newFunctionWithName:@"unordered"];
      descriptor.rasterSampleCount = 1;
      id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
      if (!pipeline) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
      for (unsigned repeat = 0; repeat < kRepeats; ++repeat) {
        id<MTLBuffer> buffer = [device newBufferWithLength:kSize * kSize * 4 options:MTLResourceStorageModeShared];
        std::fill_n(static_cast<uint32_t*>(buffer.contents), kSize * kSize, 0u);
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.renderTargetWidth = kSize;
        pass.renderTargetHeight = kSize;
        pass.defaultRasterSampleCount = 1;
        id<MTLCommandBuffer> cb = [queue commandBuffer];
        id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
        [encoder setRenderPipelineState:pipeline];
        [encoder setFragmentBuffer:top_level offset:0 atIndex:2];
        [encoder setFragmentBuffer:buffer offset:0 atIndex:8];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3 * kPrimitives];
        [encoder endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) return std::fprintf(stderr, "%s\n", cb.error.description.UTF8String), 1;
        const uint32_t* p = static_cast<const uint32_t*>(buffer.contents);
        unsigned wrong = 0;
        for (unsigned i = 0; i < kSize * kSize; ++i) wrong += p[i] != expected;
        if (arm == 2) {
          unordered_wrong += wrong;
        } else if (wrong) {
          return std::fprintf(stderr, "%s: %u of %u pixels out of primitive order\n",
                              arm ? "MSL raster_order_group" : "spirv2air", wrong, kSize * kSize),
                 1;
        }
      }
    }
    std::printf("PASS: %u x %u pixels in primitive order (%u primitives, %u passes); unordered MSL: %u wrong\n", kSize,
                kSize, kPrimitives, kRepeats, unordered_wrong);
    return 0;
  }
}
