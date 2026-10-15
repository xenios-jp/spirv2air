// Checks spirv2air's Metal function constants for xenia "fetch state" uniforms
// (Options::graphics.functionConstantSet): the fetchstate.frag shader reads three uint4 uniforms of descriptor set 9
// (bindings 0, 3 and 7); the host provides them as function constants named spirv2air_fc_<binding>.
//  - the library lists exactly those three constants (name, UInt4, index = binding, required);
//  - a function specialized with no value, or a value missing, still builds a pipeline (as Apple's compiler's output
//    does; the constant is then undefined), while the plain function from newFunctionWithName: cannot be used in one;
//  - values given by name, and by index, reach the shader, and two specializations of one library differ;
//  - a constant that is given but not declared (binding 1) is ignored.
// usage: fc_gpu <fetchstate.frag.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cstdio>
#include <fstream>
#include <vector>

namespace {

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
vertex float4 vtx(uint v [[vertex_id]]) {
  const float2 big[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  return float4(big[v], 0.5, 1);
}
)";

int failures = 0;
#define CHECK(cond, ...)                           \
  do {                                             \
    if (!(cond)) {                                 \
      fprintf(stderr, "FAIL line %d: ", __LINE__); \
      fprintf(stderr, __VA_ARGS__);                \
      fprintf(stderr, "\n");                       \
      ++failures;                                  \
    }                                              \
  } while (0)

struct U4 {
  uint32_t v[4];
};

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 2) return fprintf(stderr, "usage: fc_gpu <fetchstate.frag.spv>\n"), 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> source = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!source) return fprintf(stderr, "MSL: %s\n", error.description.UTF8String), 1;

    std::vector<uint32_t> words = readWords(argv[1]);
    spirv2air::Options options;
    options.graphics.functionConstantSet = 9;
    spirv2air::Result r = spirv2air::translate(words.data(), words.size(), options);
    if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), 1;
    // The option is off by default: the same module is rejected.
    CHECK(!spirv2air::translate(words.data(), words.size()).ok(), "translated without functionConstantSet");
    dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
    id<MTLLibrary> lib = [device newLibraryWithData:blob error:&error];
    if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), 1;

    // Reflection of the constants.
    id<MTLFunction> unspecialized = [lib newFunctionWithName:@"main"];
    if (unspecialized) {
      NSDictionary<NSString*, MTLFunctionConstant*>* fcs = unspecialized.functionConstantsDictionary;
      CHECK(fcs.count == 3, "%lu function constants", (unsigned long)fcs.count);
      for (NSNumber* binding in @[ @0, @3, @7 ]) {
        MTLFunctionConstant* c = fcs[[NSString stringWithFormat:@"spirv2air_fc_%@", binding]];
        CHECK(c, "constant for binding %lu missing", (unsigned long)binding.unsignedIntegerValue);
        if (!c) continue;
        CHECK(c.type == MTLDataTypeUInt4, "type %lu", (unsigned long)c.type);
        CHECK(c.index == binding.unsignedIntegerValue, "index %lu", (unsigned long)c.index);
        CHECK(c.required, "binding %lu not required", (unsigned long)binding.unsignedIntegerValue);
      }
    }
    printf("function without constant values: %s\n", unspecialized ? "created (deferred)" : "nil");

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:4
                                                                                 height:4
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> target = [device newTextureWithDescriptor:td];
    id<MTLBuffer> tlab = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];  // unused arguments

    // Renders `fragment` and returns pixel (1, 2). `pipelineError` receives a pipeline creation failure.
    auto pipelineFor = [&](id<MTLFunction> fragment, NSError** e) -> id<MTLRenderPipelineState> {
      MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = [source newFunctionWithName:@"vtx"];
      d.fragmentFunction = fragment;
      d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
      return [device newRenderPipelineStateWithDescriptor:d error:e];
    };
    auto render = [&](MTLFunctionConstantValues* values, float out[4]) -> bool {
      NSError* e = nil;
      id<MTLFunction> fragment = [lib newFunctionWithName:@"main" constantValues:values error:&e];
      if (!fragment)
        return fprintf(stderr, "newFunctionWithName:constantValues: %s\n", e.description.UTF8String), false;
      id<MTLRenderPipelineState> pso = pipelineFor(fragment, &e);
      if (!pso) return fprintf(stderr, "pipeline: %s\n", e.description.UTF8String), false;
      MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = target;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:pso];
      [enc setFragmentBuffer:tlab offset:0 atIndex:2];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return fprintf(stderr, "GPU: %s\n", cb.error.description.UTF8String), false;
      [target getBytes:out bytesPerRow:4 * 16 fromRegion:MTLRegionMake2D(1, 2, 1, 1) mipmapLevel:0];
      return true;
    };
    auto setByName = [](MTLFunctionConstantValues* v, int binding, U4 value) {
      NSString* name = [NSString stringWithFormat:@"spirv2air_fc_%d", binding];
      [v setConstantValue:&value type:MTLDataTypeUInt4 withName:name];
    };

    float px[4];
    // Missing values are not an error: as for a function Apple's own compiler builds from MSL (checked while writing
    // this test), Metal creates the function and the pipeline, and the constants are simply not defined. Only the
    // plain function from newFunctionWithName: cannot be used in a pipeline (Metal asserts), so it is not tried.
    {
      NSError* e = nil;
      id<MTLFunction> f = [lib newFunctionWithName:@"main" constantValues:[MTLFunctionConstantValues new] error:&e];
      printf("no values: function %s", f ? "created" : e.description.UTF8String);
      if (f) {
        id<MTLRenderPipelineState> pso = pipelineFor(f, &e);
        CHECK(pso, "no pipeline with no constant values: %s", e.description.UTF8String);
        printf(", pipeline %s\n", pso ? "built" : e.description.UTF8String);
      }
      MTLFunctionConstantValues* v = [MTLFunctionConstantValues new];
      setByName(v, 0, {{1, 2, 3, 4}});
      setByName(v, 3, {{1, 2, 3, 4}});
      f = [lib newFunctionWithName:@"main" constantValues:v error:&e];
      printf("one value missing: function %s", f ? "created" : e.description.UTF8String);
      if (f) {
        id<MTLRenderPipelineState> pso = pipelineFor(f, &e);
        CHECK(pso, "no pipeline with a missing constant value: %s", e.description.UTF8String);
        printf(", pipeline %s\n", pso ? "built" : e.description.UTF8String);
      }
    }
    // Values by name; binding 1 is not declared and is ignored.
    {
      MTLFunctionConstantValues* v = [MTLFunctionConstantValues new];
      setByName(v, 0, {{11, 12, 13, 14}});
      setByName(v, 1, {{91, 92, 93, 94}});
      setByName(v, 3, {{21, 22, 23, 24}});
      setByName(v, 7, {{31, 32, 33, 34}});
      bool ok = render(v, px);
      CHECK(ok, "render by name");
      if (ok)
        CHECK(px[0] == 11 && px[1] == 14 && px[2] == 23 && px[3] == 32, "by name: %g %g %g %g", px[0], px[1], px[2],
              px[3]);
    }
    // A second specialization of the same library, by index.
    {
      MTLFunctionConstantValues* v = [MTLFunctionConstantValues new];
      U4 a = {{101, 102, 103, 104}}, b = {{201, 202, 203, 204}}, c = {{301, 302, 303, 304}};
      [v setConstantValue:&a type:MTLDataTypeUInt4 atIndex:0];
      [v setConstantValue:&b type:MTLDataTypeUInt4 atIndex:3];
      [v setConstantValue:&c type:MTLDataTypeUInt4 atIndex:7];
      bool ok = render(v, px);
      CHECK(ok, "render by index");
      if (ok)
        CHECK(px[0] == 101 && px[1] == 104 && px[2] == 203 && px[3] == 302, "by index: %g %g %g %g", px[0], px[1],
              px[2], px[3]);
    }
    // Large and zero values.
    {
      MTLFunctionConstantValues* v = [MTLFunctionConstantValues new];
      setByName(v, 0, {{0, 0, 0, 16777216}});
      setByName(v, 3, {{0, 0, 0, 0}});
      setByName(v, 7, {{0, 7, 0, 0}});
      bool ok = render(v, px);
      CHECK(ok, "render zeros");
      if (ok)
        CHECK(px[0] == 0 && px[1] == 16777216 && px[2] == 0 && px[3] == 7, "zeros: %g %g %g %g", px[0], px[1], px[2],
              px[3]);
    }
    printf(failures ? "fc_gpu: %d FAILED\n" : "fc_gpu: all passed\n", failures);
    return failures ? 1 : 0;
  }
}
