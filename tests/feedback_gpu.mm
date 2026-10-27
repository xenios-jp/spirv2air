// Ordered attachment feedback must preserve every intra-draw update per sample.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <spirv2air.h>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <vector>

int main(int argc, char** argv) { @autoreleasepool {
  if (argc != 2) return 2;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device || !device.rasterOrderGroupsSupported) return 77;
  printf("Device: %s\n", device.name.UTF8String);
  std::ifstream file(argv[1], std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(file ? size_t(file.tellg()) / 4 : 0);
  if (words.size() < 5) return 2;
  file.seekg(0);
  file.read(reinterpret_cast<char*>(words.data()), words.size() * 4);
  NSString* source = @"#include <metal_stdlib>\nusing namespace metal;\n"
    "vertex float4 vs(uint v [[vertex_id]]) { const float2 p[3]={float2(-1,-1),float2(3,-1),float2(-1,3)}; return float4(p[v%3],0,1); }\n"
    "struct O {float4 c [[color(0),raster_order_group(0)]];};\n"
    "fragment O fs(float4 prev [[color(0),raster_order_group(0)]],uint s [[sample_id]]) {return {prev+float4(.25,.125,float(s+1)*.0625,.25)};}\n";
  NSError* error = nil;
  id<MTLLibrary> reference = [device newLibraryWithSource:source options:nil error:&error];
  if (!reference) { puts(error.description.UTF8String); return 1; }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  for (bool readOnlySampleId : {false, true}) {
    auto input = words;
    if (readOnlySampleId) {
      // Leave SampleId in OpImageRead only, to test sample-rate execution even
      // when AIR arithmetic no longer consumes that argument. Blue adds 1/64.
      uint32_t quarter = 0;
      for (size_t p = 5; p < input.size(); p += input[p] >> 16)
        if ((input[p] & 65535) == 43 && input[p + 3] == 0x3E800000) quarter = input[p + 2];
      if (!quarter) return 2;
      for (size_t p = 5; p < input.size(); p += input[p] >> 16)
        if ((input[p] & 65535) == 133) input[p + 3] = quarter; // OpFMul
    }
    spirv2air::Options options;
    options.graphics.framebufferFetchMask = 1;
    auto result = spirv2air::translate(input.data(), input.size(), options);
    if (!result.ok()) { puts(result.error.c_str()); return 1; }
    dispatch_data_t data = dispatch_data_create(result.metallib.data(), result.metallib.size(),
                                                nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> translated = [device newLibraryWithData:data error:&error];
    if (!translated) { puts(error.description.UTF8String); return 1; }
    for (NSUInteger samples : {1ul, 2ul, 4ul}) {
      if (![device supportsTextureSampleCount:samples]) continue;
      MTLRenderPipelineDescriptor* pipelineDesc = [MTLRenderPipelineDescriptor new];
      pipelineDesc.vertexFunction = [reference newFunctionWithName:@"vs"];
      pipelineDesc.fragmentFunction = [translated newFunctionWithName:@"main"];
      pipelineDesc.rasterSampleCount = samples;
      pipelineDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
      pipelineDesc.colorAttachments[0].blendingEnabled = NO;
      id<MTLRenderPipelineState> pipeline = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
      if (!pipeline) { puts(error.description.UTF8String); return 1; }
      pipelineDesc.fragmentFunction = [reference newFunctionWithName:@"fs"];
      id<MTLRenderPipelineState> seed = [device newRenderPipelineStateWithDescriptor:pipelineDesc error:&error];
      if (!seed) { puts(error.description.UTF8String); return 1; }
      MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                 width:16 height:16 mipmapped:NO];
      td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
      td.storageMode = MTLStorageModeShared;
      id<MTLTexture> readback = [device newTextureWithDescriptor:td], target = readback;
      if (samples > 1) {
        td.storageMode = MTLStorageModePrivate;
        td.textureType = MTLTextureType2DMultisample;
        td.sampleCount = samples;
        target = [device newTextureWithDescriptor:td];
      }
      MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = target;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
      pass.colorAttachments[0].storeAction = samples > 1 ? MTLStoreActionMultisampleResolve : MTLStoreActionStore;
      if (samples > 1) pass.colorAttachments[0].resolveTexture = readback;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
      if (readOnlySampleId) {
        // Each sample starts with a distinct value. Broadcasting sample zero's
        // attachment to the other samples must fail the resolved comparison.
        [enc setRenderPipelineState:seed];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      }
      [enc setRenderPipelineState:pipeline];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:300];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:60];
      [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError) { puts(cb.error.description.UTF8String); return 1; }
      _Float16 values[16 * 16 * 4];
      [readback getBytes:values bytesPerRow:16 * 8 fromRegion:MTLRegionMake2D(0, 0, 16, 16) mipmapLevel:0];
      float expected[4] = {30, 15, 7.5f * (samples + 1) / 2, 30};
      if (readOnlySampleId) {
        expected[0] += .25f; expected[1] += .125f;
        expected[2] = 1.875f + .0625f * (samples + 1) / 2; expected[3] += .25f;
      }
      unsigned wrong = 0;
      for (unsigned p = 0; p < 256; ++p)
        for (unsigned c = 0; c < 4; ++c) wrong += float(values[4 * p + c]) != expected[c];
      printf("samples=%lu read-only SampleId=%d wrong=%u/1024\n", samples, readOnlySampleId, wrong);
      if (wrong) return 1;
    }
  }
  return 0;
}}
