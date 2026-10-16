// Compare the supported linear primitive barycentric input with native MSL over
// projected triangles, at native single-sample and four-sample rasterization.
#import <Metal/Metal.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <spirv2air.h>
#include <vector>

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !device.supportsShaderBarycentricCoordinates) return 77;
    std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
    std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
    if (words.size() < 5) return 2;
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
    NSString* source = @R"(
      #include <metal_stdlib>
      using namespace metal;
      vertex float4 vertex_main(uint id [[vertex_id]]) {
        const float2 xy[4] = {float2(-0.75,-0.75),float2(0.75,-0.75),float2(-0.75,0.75),float2(0.75,0.75)};
        const float w[4] = {0.5,2.0,1.0,0.75};
        return float4(xy[id]*w[id],0.25*w[id],w[id]);
      }
      struct Linear {float3 bc [[barycentric_coord,center_no_perspective]];};
      fragment float4 reference_linear(Linear v [[stage_in]]) {return float4(v.bc,1);}
      kernel void readback(texture2d_ms<float,access::read> t [[texture(0)]],device float4* out [[buffer(0)]],
                           uint3 id [[thread_position_in_grid]]) {out[(id.y*16+id.x)*4+id.z]=t.read(id.xy,id.z);}
    )";
    NSError* error = nil;
    id<MTLLibrary> reference = [device newLibraryWithSource:source options:nil error:&error];
    if (!reference) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLComputePipelineState> reader =
        [device newComputePipelineStateWithFunction:[reference newFunctionWithName:@"readback"] error:&error];
    if (!reader) return 1;
    size_t compared = 0;
    {
      auto input = words;
      auto result = spirv2air::translate(input.data(), input.size());
      if (!result.ok()) return std::fprintf(stderr, "%s\n", result.error.c_str()), 1;
      dispatch_data_t blob = dispatch_data_create(result.metallib.data(), result.metallib.size(), nullptr, nullptr);
      id<MTLLibrary> translated = [device newLibraryWithData:blob error:&error];
      if (!translated) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
      for (unsigned samples : {1u, 4u}) {
        std::vector<std::vector<uint32_t>> outputs;
        for (unsigned arm = 0; arm < 2; ++arm) {
          MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
          descriptor.vertexFunction = [reference newFunctionWithName:@"vertex_main"];
          descriptor.fragmentFunction =
              arm ? [translated newFunctionWithName:@"main"] : [reference newFunctionWithName:@"reference_linear"];
          descriptor.rasterSampleCount = samples;
          descriptor.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
          id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
          if (!pipeline) return std::fprintf(stderr, "%s\n", error.description.UTF8String), 1;
          MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                        width:16
                                                                                       height:16
                                                                                    mipmapped:NO];
          td.sampleCount = samples;
          td.textureType = samples == 1 ? MTLTextureType2D : MTLTextureType2DMultisample;
          td.storageMode = MTLStorageModePrivate;
          td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
          id<MTLTexture> texture = [device newTextureWithDescriptor:td];
          MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
          pass.colorAttachments[0].texture = texture;
          pass.colorAttachments[0].loadAction = MTLLoadActionClear;
          pass.colorAttachments[0].storeAction = MTLStoreActionStore;
          id<MTLCommandBuffer> cb = [queue commandBuffer];
          id<MTLRenderCommandEncoder> encoder = [cb renderCommandEncoderWithDescriptor:pass];
          [encoder setRenderPipelineState:pipeline];
          [encoder drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
          [encoder endEncoding];
          NSUInteger bytes = 16 * 16 * samples * 16;
          id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
          if (samples == 1) {
            id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
            [blit copyFromTexture:texture
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:MTLOriginMake(0, 0, 0)
                              sourceSize:MTLSizeMake(16, 16, 1)
                                toBuffer:buffer
                       destinationOffset:0
                  destinationBytesPerRow:256
                destinationBytesPerImage:4096];
            [blit endEncoding];
          } else {
            id<MTLComputeCommandEncoder> compute = [cb computeCommandEncoder];
            [compute setComputePipelineState:reader];
            [compute setTexture:texture atIndex:0];
            [compute setBuffer:buffer offset:0 atIndex:0];
            [compute dispatchThreads:MTLSizeMake(16, 16, 4) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [compute endEncoding];
          }
          [cb commit];
          [cb waitUntilCompleted];
          if (cb.error) return std::fprintf(stderr, "%s\n", cb.error.description.UTF8String), 1;
          const uint32_t* p = static_cast<const uint32_t*>(buffer.contents);
          outputs.emplace_back(p, p + bytes / 4);
        }
        if (outputs[0] != outputs[1]) return std::fprintf(stderr, "%uX barycentric mismatch\n", samples), 1;
        // Pixel center (4.5, 8.5) is inside triangle 0. The integer screen
        // vertices are (2,14), (14,14), (2,2), so these weights independently
        // check the SPIR-V vertex order independently of the MSL comparison.
        const float expected[3] = {1.0f / 3, 5.0f / 24, 11.0f / 24};
        for (unsigned sample = 0; sample < samples; ++sample)
          for (unsigned c = 0; c < 3; ++c) {
            float actual;
            uint32_t word = outputs[1][((8 * 16 + 4) * samples + sample) * 4 + c];
            std::memcpy(&actual, &word, sizeof(actual));
            if (std::fabs(actual - expected[c]) > 0.00002f)
              return std::fprintf(stderr, "barycentric mismatch samples%u component%u actual%.9g expected%.9g\n",
                                  samples, c, actual, expected[c]),
                     1;
          }
        bool covered = false;
        for (size_t i = 3; i < outputs[0].size(); i += 4)
          covered |= outputs[0][i] == 0x3f800000;
        if (!covered) return std::fprintf(stderr, "empty barycentric output\n"), 1;
        compared += outputs[0].size();
      }
    }
    std::printf("PASS: %zu barycentric component words match MSL at 1X and 4X\n", compared);
    return 0;
  }
}
