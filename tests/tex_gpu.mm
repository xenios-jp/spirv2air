// Checks spirv2air's texture sampling on the GPU: the handle type and sample intrinsic must match the view type the
// host binds, and the array layer must follow SPIR-V's clamp(RoundEven(coordinate), 0, layers - 1). Each translated
// pixel shader (tests/data/tex/*.frag.spvasm) samples a texture of the matching MTLTextureType (2D array, 2D, cube,
// cube array, 3D) through the graphics heap/argument-buffer layout. Cube, cube array, 2D and 3D results are compared
// bit for bit with a hand-written MSL reference sampling the same texture; every slice holds a distinct constant so the
// layer or face that was sampled is identified by the value. layer2da.frag: layers -3, -0.5, 0.5, 1.5, 2.5, 3.4, 7,
// 1e20, NaN and -0 of a 4-layer array. Round-half-even gives layers 0, 0, 0, 2, 2, 3 and the clamp gives 3 for 7 and
// 1e20; NaN is 0 (as D3D12, where SPIR-V leaves it open). usage: tex_gpu <layer2da.spv> <tex2d.spv> <cube.spv>
// <cubearray.spv> <tex3d.spv> <layer2dagrad.spv> <cubegrad.spv> <tex3dgrad.spv> (the last three repeat the array, cube
// and 3D lookups with explicit zero gradients: the _grad forms of the sample intrinsics) <lodbias.spv> (the Bias and
// Lod operands on a 4-level mipmapped 2D texture whose level m holds m + 1, against MSL bias() and level())
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

namespace {

constexpr int kResourceHeap = 0, kSamplerHeap = 1, kTlab = 2;
constexpr int kEntry = 24;            // heap record size
constexpr int kFragmentIndices = 72;  // TLAB offset of the fragment texture index buffer

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
  const float2 p[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
  return float4(p[v], 0.5, 1);
}
fragment float4 ref_2d(float4 pos [[position]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                       constant float4& c [[buffer(3)]]) {
  return t.sample(s, c.xy, level(0));
}
fragment float4 ref_lodbias(float4 pos [[position]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                            constant float4& c [[buffer(3)]]) {
  float2 uv = pos.xy * c.x;
  return float4(t.sample(s, uv, bias(c.y)).x, t.sample(s, uv, level(c.z)).x, 0, 1);
}
fragment float4 ref_cube(float4 pos [[position]], texturecube<float> t [[texture(0)]], sampler s [[sampler(0)]],
                         constant float4& c [[buffer(3)]]) {
  return t.sample(s, c.xyz, level(0));
}
fragment float4 ref_cubearray(float4 pos [[position]], texturecube_array<float> t [[texture(0)]],
                              sampler s [[sampler(0)]], constant float4& c [[buffer(3)]]) {
  return t.sample(s, c.xyz, uint(c.w), level(0));
}
fragment float4 ref_3d(float4 pos [[position]], texture3d<float> t [[texture(0)]], sampler s [[sampler(0)]],
                       constant float4& c [[buffer(3)]]) {
  return t.sample(s, c.xyz, level(0));
}
)";

// A texture of `type` whose slice k (array layer, cube face, or depth plane) is the constant (k + 1, 0.25, 0.5, 1).
id<MTLTexture> makeTexture(id<MTLDevice> device, MTLTextureType type, int slices) {
  MTLTextureDescriptor* d = [MTLTextureDescriptor new];
  d.textureType = type;
  d.pixelFormat = MTLPixelFormatRGBA32Float;
  d.width = 4;
  d.height = 4;
  d.depth = type == MTLTextureType3D ? slices : 1;
  d.arrayLength = (type == MTLTextureType2DArray || type == MTLTextureTypeCubeArray)
                      ? (type == MTLTextureTypeCubeArray ? slices / 6 : slices)
                      : 1;
  d.usage = MTLTextureUsageShaderRead;
  d.storageMode = MTLStorageModeShared;
  id<MTLTexture> t = [device newTextureWithDescriptor:d];
  for (int k = 0; k < slices; ++k) {
    float texels[16][4];
    for (auto& texel : texels) texel[0] = float(k + 1), texel[1] = 0.25f, texel[2] = 0.5f, texel[3] = 1.0f;
    if (type == MTLTextureType3D)
      [t replaceRegion:MTLRegionMake3D(0, 0, k, 4, 4, 1)
            mipmapLevel:0
                  slice:0
              withBytes:texels
            bytesPerRow:64
          bytesPerImage:256];
    else
      [t replaceRegion:MTLRegionMake2D(0, 0, 4, 4)
            mipmapLevel:0
                  slice:k
              withBytes:texels
            bytesPerRow:64
          bytesPerImage:256];
  }
  return t;
}

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 10) {
      fprintf(stderr, "usage: tex_gpu <layer2da.spv> <tex2d.spv> <cube.spv> <cubearray.spv> <tex3d.spv> ");
      return fprintf(stderr, "<layer2dagrad.spv> <cubegrad.spv> <tex3dgrad.spv> <lodbias.spv>\n"), 2;
    }
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 1;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    NSError* error = nil;
    id<MTLLibrary> source = [device newLibraryWithSource:kSource options:nil error:&error];
    if (!source) return fprintf(stderr, "MSL: %s\n", error.description.UTF8String), 1;
    auto load = [&](const char* path) -> id<MTLFunction> {
      std::vector<uint32_t> words = readWords(path);
      spirv2air::Result r = spirv2air::translate(words.data(), words.size());
      if (!r.ok()) return fprintf(stderr, "translate %s failed: %s\n", path, r.error.c_str()), nil;
      dispatch_data_t blob = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
      NSError* e = nil;
      id<MTLLibrary> lib = [device newLibraryWithData:blob error:&e];
      if (!lib) return fprintf(stderr, "newLibraryWithData: %s\n", e.description.UTF8String), nil;
      return [lib newFunctionWithName:@"main"];
    };
    id<MTLFunction> layerFn = load(argv[1]), twoDFn = load(argv[2]), cubeFn = load(argv[3]),
                    cubeArrayFn = load(argv[4]), threeDFn = load(argv[5]), layerGradFn = load(argv[6]),
                    cubeGradFn = load(argv[7]), threeDGradFn = load(argv[8]), lodBiasFn = load(argv[9]);
    if (!lodBiasFn || !layerFn || !twoDFn || !cubeFn || !cubeArrayFn || !threeDFn || !layerGradFn || !cubeGradFn ||
        !threeDGradFn)
      return 1;

    auto pipeline = [&](id<MTLFunction> fragment) -> id<MTLRenderPipelineState> {
      MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = [source newFunctionWithName:@"vtx"];
      d.fragmentFunction = fragment;
      d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA32Float;
      NSError* e = nil;
      id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:d error:&e];
      if (!p) fprintf(stderr, "pipeline: %s\n", e.description.UTF8String);
      return p;
    };
    id<MTLRenderPipelineState> layerState = pipeline(layerFn), twoDState = pipeline(twoDFn),
                               cubeState = pipeline(cubeFn), cubeArrayState = pipeline(cubeArrayFn),
                               threeDState = pipeline(threeDFn), layerGradState = pipeline(layerGradFn),
                               cubeGradState = pipeline(cubeGradFn), threeDGradState = pipeline(threeDGradFn),
                               lodBiasState = pipeline(lodBiasFn),
                               refLodBias = pipeline([source newFunctionWithName:@"ref_lodbias"]);
    id<MTLRenderPipelineState> ref2d = pipeline([source newFunctionWithName:@"ref_2d"]),
                               refCube = pipeline([source newFunctionWithName:@"ref_cube"]),
                               refCubeArray = pipeline([source newFunctionWithName:@"ref_cubearray"]),
                               ref3d = pipeline([source newFunctionWithName:@"ref_3d"]);
    if (!lodBiasState || !refLodBias || !layerGradState || !cubeGradState || !threeDGradState || !layerState ||
        !twoDState || !cubeState || !cubeArrayState || !threeDState || !ref2d || !refCube || !refCubeArray || !ref3d)
      return 1;

    MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
    sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterNearest;
    sd.sAddressMode = sd.tAddressMode = sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
    sd.supportArgumentBuffers = YES;
    id<MTLSamplerState> sampler = [device newSamplerStateWithDescriptor:sd];

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:2
                                                                                 height:2
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModeShared;

    // Draws a full-target triangle with the translated shader (texture through the heaps) and returns pixel (0, 0).
    auto drawTranslated = [&](id<MTLRenderPipelineState> state, id<MTLTexture> texture, const float coord[4],
                              float out[4]) {
      id<MTLBuffer> ubo = [device newBufferWithBytes:coord length:16 options:MTLResourceStorageModeShared];
      id<MTLBuffer> resourceHeap = [device newBufferWithLength:kEntry options:MTLResourceStorageModeShared];
      id<MTLBuffer> samplerHeap = [device newBufferWithLength:kEntry options:MTLResourceStorageModeShared];
      id<MTLBuffer> indices = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
      id<MTLBuffer> tlab = [device newBufferWithLength:112 options:MTLResourceStorageModeShared];
      memset(resourceHeap.contents, 0, kEntry);
      memset(samplerHeap.contents, 0, kEntry);
      memset(indices.contents, 0, 16);  // binding 0: texture heap index 0; binding 1: sampler heap index 0
      memset(tlab.contents, 0, 112);
      MTLResourceID textureId = texture.gpuResourceID, samplerId = sampler.gpuResourceID;
      memcpy(static_cast<char*>(resourceHeap.contents) + 8, &textureId, 8);
      memcpy(samplerHeap.contents, &samplerId, 8);
      uint64_t uboAddress = ubo.gpuAddress, indexAddress = indices.gpuAddress;
      memcpy(tlab.contents, &uboAddress, 8);
      memcpy(static_cast<char*>(tlab.contents) + kFragmentIndices, &indexAddress, 8);
      id<MTLTexture> target = [device newTextureWithDescriptor:td];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = target;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:state];
      [enc setFragmentBuffer:resourceHeap offset:0 atIndex:kResourceHeap];
      [enc setFragmentBuffer:samplerHeap offset:0 atIndex:kSamplerHeap];
      [enc setFragmentBuffer:tlab offset:0 atIndex:kTlab];
      [enc useResource:texture usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
      [enc useResource:ubo usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
      [enc useResource:indices usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), false;
      [target getBytes:out bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
      return true;
    };
    auto drawReference = [&](id<MTLRenderPipelineState> state, id<MTLTexture> texture, const float coord[4],
                             float out[4]) {
      id<MTLTexture> target = [device newTextureWithDescriptor:td];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
      pass.colorAttachments[0].texture = target;
      pass.colorAttachments[0].loadAction = MTLLoadActionClear;
      pass.colorAttachments[0].clearColor = MTLClearColorMake(-1, -1, -1, -1);
      pass.colorAttachments[0].storeAction = MTLStoreActionStore;
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:state];
      [enc setFragmentTexture:texture atIndex:0];
      [enc setFragmentSamplerState:sampler atIndex:0];
      [enc setFragmentBytes:coord length:16 atIndex:3];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), false;
      [target getBytes:out bytesPerRow:32 fromRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0];
      return true;
    };

    int failures = 0;
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // Array layer rule: the expected value of the first channel is the layer + 1.
    id<MTLTexture> array = makeTexture(device, MTLTextureType2DArray, 4);
    struct LayerCase {
      float z;
      int layer;
    };
    const LayerCase layers[] = {{-3, 0}, {-0.5f, 0}, {0.5f, 0}, {1.5f, 2},  {2.5f, 2}, {3.4f, 3},
                                {7, 3},  {1e20f, 3}, {nan, 0},  {-0.0f, 0}, {0, 0},    {1, 1}};
    for (const LayerCase& c : layers) {
      float coord[4] = {0.5f, 0.5f, c.z, 0}, got[4];
      if (!drawTranslated(layerState, array, coord, got)) return 1;
      bool ok = got[0] == float(c.layer + 1) && got[1] == 0.25f;
      printf("tex_gpu: array layer z=%g -> %g (want %d)%s\n", c.z, got[0] - 1, c.layer, ok ? "" : "  FAIL");
      failures += !ok;
    }

    // The other view types: bit-identical to the MSL reference, and the slice is the expected one.
    auto compare = [&](const char* what, id<MTLRenderPipelineState> state, id<MTLRenderPipelineState> reference,
                       id<MTLTexture> texture, const float coord[4], int wantSlice) {
      float got[4], want[4];
      if (!drawTranslated(state, texture, coord, got) || !drawReference(reference, texture, coord, want)) return false;
      bool ok = memcmp(got, want, 16) == 0 && got[0] == float(wantSlice + 1);
      printf("tex_gpu: %s (%g %g %g %g) -> slice %g, reference slice %g, want %d%s\n", what, coord[0], coord[1],
             coord[2], coord[3], got[0] - 1, want[0] - 1, wantSlice, ok ? "" : "  FAIL");
      failures += !ok;
      return true;
    };
    id<MTLTexture> plain = makeTexture(device, MTLTextureType2D, 1);
    const float uv[4] = {0.3f, 0.6f, 0, 0};
    if (!compare("2D", twoDState, ref2d, plain, uv, 0)) return 1;

    id<MTLTexture> cube = makeTexture(device, MTLTextureTypeCube, 6);
    id<MTLTexture> cubeArray = makeTexture(device, MTLTextureTypeCubeArray, 12);
    // Metal's face order: +X, -X, +Y, -Y, +Z, -Z.
    const float dirs[6][4] = {{1, 0.2f, -0.1f, 0},  {-1, 0.1f, 0.3f, 0}, {0.1f, 1, 0.2f, 0},
                              {-0.2f, -1, 0.1f, 0}, {0.3f, 0.1f, 1, 0},  {0.1f, -0.3f, -1, 0}};
    for (int face = 0; face < 6; ++face) {
      if (!compare("cube", cubeState, refCube, cube, dirs[face], face)) return 1;
      for (float layer : {0.0f, 1.0f, 1.4f, -5.0f, 1e20f}) {
        float coord[4] = {dirs[face][0], dirs[face][1], dirs[face][2], layer};
        int wantLayer = layer < 0.5f ? 0 : 1;  // the array has 2 cubes: the layer clamps to the last one
        if (!compare("cube array", cubeArrayState, refCubeArray, cubeArray, coord, wantLayer * 6 + face)) return 1;
      }
    }

    id<MTLTexture> volume = makeTexture(device, MTLTextureType3D, 4);
    for (int k = 0; k < 4; ++k) {
      const float coord[4] = {0.5f, 0.5f, (k + 0.5f) / 4, 0};
      if (!compare("3D", threeDState, ref3d, volume, coord, k)) return 1;
    }

    // The explicit-gradient forms: zero gradients select level 0, so the results are those of the lookups above.
    for (const LayerCase& c : {LayerCase{-3, 0}, LayerCase{1.5f, 2}, LayerCase{1e20f, 3}, LayerCase{nan, 0}}) {
      float coord[4] = {0.5f, 0.5f, c.z, 0}, got[4];
      if (!drawTranslated(layerGradState, array, coord, got)) return 1;
      bool ok = got[0] == float(c.layer + 1) && got[1] == 0.25f;
      printf("tex_gpu: array layer with gradients z=%g -> %g (want %d)%s\n", c.z, got[0] - 1, c.layer,
             ok ? "" : "  FAIL");
      failures += !ok;
    }
    for (int face = 0; face < 6; ++face)
      if (!compare("cube with gradients", cubeGradState, refCube, cube, dirs[face], face)) return 1;
    for (int k = 0; k < 4; ++k) {
      const float coord[4] = {0.5f, 0.5f, (k + 0.5f) / 4, 0};
      if (!compare("3D with gradients", threeDGradState, ref3d, volume, coord, k)) return 1;
    }

    // Bias and Lod: level m of an 8x8 texture holds m + 1. uv = pixel position / 8 makes one texel per pixel at level
    // 0, so the implicit level is 0 and the biased level is the bias (clamped to 0..3); the explicit level is c.z.
    MTLTextureDescriptor* md = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                  width:8
                                                                                 height:8
                                                                              mipmapped:YES];
    md.usage = MTLTextureUsageShaderRead;
    md.storageMode = MTLStorageModeShared;
    id<MTLTexture> mipped = [device newTextureWithDescriptor:md];
    for (NSUInteger level = 0; level < mipped.mipmapLevelCount; ++level) {
      int n = 8 >> level;
      std::vector<float> texels(size_t(n * n * 4), float(level + 1));
      [mipped replaceRegion:MTLRegionMake2D(0, 0, n, n) mipmapLevel:level withBytes:texels.data() bytesPerRow:n * 16];
    }
    sd.mipFilter = MTLSamplerMipFilterNearest;
    sampler = [device newSamplerStateWithDescriptor:sd];
    struct LodCase {
      float bias, lod;
      int biasLevel, lodLevel;  // -1: compared with the reference only
    };
    for (const LodCase& c : {LodCase{0, 0, 0, 0}, LodCase{1, 1, 1, 1}, LodCase{2, 2, 2, 2}, LodCase{3, 3, 3, 3},
                             LodCase{10, 7, 3, 3}, LodCase{-3, -1, 0, 0}, LodCase{1.3f, 2.7f, -1, -1}}) {
      const float coord[4] = {0.125f, c.bias, c.lod, 0};
      float got[4], want[4];
      if (!drawTranslated(lodBiasState, mipped, coord, got) || !drawReference(refLodBias, mipped, coord, want))
        return 1;
      bool ok = memcmp(got, want, 16) == 0 && (c.biasLevel < 0 || got[0] == float(c.biasLevel + 1)) &&
                (c.lodLevel < 0 || got[1] == float(c.lodLevel + 1));
      printf("tex_gpu: bias %g -> level %g (reference %g), lod %g -> level %g (reference %g)%s\n", c.bias, got[0] - 1,
             want[0] - 1, c.lod, got[1] - 1, want[1] - 1, ok ? "" : "  FAIL");
      failures += !ok;
    }
    printf("tex_gpu: %d failures\n", failures);
    return failures ? 1 : 0;
  }
}
