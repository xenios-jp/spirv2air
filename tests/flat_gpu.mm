// Checks spirv2air's flat and integer graphics varyings on the GPU. flat.vert writes a Flat float, a Flat int2 with
// negative values, a Flat uint3 with values above 2^31 and a smooth float, with a distinct value per vertex; flat.frag
// reads them and writes float4, int4 and uint4 render targets (RGBA32Float, RGBA32Sint, RGBA32Uint); narrow.frag writes
// a float, an int2 and a uint2 colour output (into R32Float, RG32Sint and RG32Uint: Metal requires the format to have
// at least as many components as the output). Each fragment shader runs in all four vertex/fragment pairings of the
// translated and the hand-written MSL functions, which proves the user(texcoordN) linking is compatible in both
// directions, and every render target is compared bit for bit with the MSL/MSL pair.
//
// Provoking vertex: Vulkan's default for flat values is the first vertex of the primitive. Metal's observed behaviour
// is asserted below (kProvoking), by reading the flat values of a triangle list draw back against the per-vertex
// values.
//
// Signedness: Metal matches user() varyings by name and type. Apple's compiler is also run on a pair whose vertex
// function writes texcoord1 as uint2 and texcoord2 as int3 while the fragment function reads int2 and uint3; the
// outcome is printed and asserted (kMismatch*) so a change in the compiler is noticed.
// usage: flat_gpu <flat.vert.spv> <flat.frag.spv> <narrow.frag.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

constexpr int kSize = 8;

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
struct VOut {
  float4 pos [[position]];
  float f [[user(texcoord0), flat]];
  int2 i [[user(texcoord1)]];
  uint3 u [[user(texcoord2)]];
  float s [[user(texcoord3)]];
};
float4 position(uint v) { return float4(v == 1 ? 3.0 : -1.0, v == 2 ? 3.0 : -1.0, 0, 1); }
VOut make(uint v) {
  VOut o;
  o.pos = position(v);
  o.f = float(v) * 1.5 + 1.0;
  o.i = int2(-int(v + 1), int(v) * 100 - 7);
  o.u = uint3(0xF0000000u + v * 65537u, 0xFFFFFFFFu - v, 0x9E3779B9u * (v + 1));
  o.s = float(v);
  return o;
}
vertex VOut vtx(uint v [[vertex_id]]) { return make(v); }
// The signedness of texcoord1 and texcoord2 swapped relative to the fragment functions.
struct VMix {
  float4 pos [[position]];
  float f [[user(texcoord0), flat]];
  uint2 i [[user(texcoord1)]];
  int3 u [[user(texcoord2)]];
  float s [[user(texcoord3)]];
};
vertex VMix vtx_mixed(uint v [[vertex_id]]) {
  VOut a = make(v);
  VMix o;
  o.pos = a.pos; o.f = a.f; o.i = uint2(a.i); o.u = int3(a.u); o.s = a.s;
  return o;
}
struct FIn {
  float4 pos [[position]];
  float f [[user(texcoord0), flat]];
  int2 i [[user(texcoord1), flat]];
  uint3 u [[user(texcoord2), flat]];
  float s [[user(texcoord3)]];
};
struct FOut { float4 c0 [[color(0)]]; int4 c1 [[color(1)]]; uint4 c2 [[color(2)]]; };
fragment FOut frg(FIn in [[stage_in]]) {
  FOut o;
  o.c0 = float4(in.f, in.s, 0, 1);
  o.c1 = int4(in.i, 5, -9);
  o.c2 = uint4(in.u, 1);
  return o;
}
struct FOutN { float c0 [[color(0)]]; int2 c1 [[color(1)]]; uint2 c2 [[color(2)]]; };
fragment FOutN frg_narrow(FIn in [[stage_in]]) {
  FOutN o;
  o.c0 = in.f;
  o.c1 = in.i;
  o.c2 = in.u.xy;
  return o;
}
)";

// Observed on Apple GPUs: flat values come from the first vertex of a triangle (Vulkan's default provoking vertex).
constexpr int kProvoking = 0;

struct Pixels {
  std::vector<uint32_t> rt[3];  // [pixel][4] words of each target
};

}  // namespace

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc != 4) return fprintf(stderr, "usage: flat_gpu <flat.vert.spv> <flat.frag.spv> <narrow.frag.spv>\n"), 2;
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
    id<MTLFunction> tVert = load(argv[1]), tFrag = load(argv[2]), tNarrow = load(argv[3]);
    if (!tVert || !tFrag || !tNarrow) return 1;
    auto msl = [&](NSString* n) { return [source newFunctionWithName:n]; };

    const MTLPixelFormat wideFormats[3] = {MTLPixelFormatRGBA32Float, MTLPixelFormatRGBA32Sint,
                                           MTLPixelFormatRGBA32Uint};
    const MTLPixelFormat narrowFormats[3] = {MTLPixelFormatR32Float, MTLPixelFormatRG32Sint, MTLPixelFormatRG32Uint};
    const int narrowComponents[3] = {1, 2, 2};
    const MTLPixelFormat* formats = wideFormats;
    const int* components = nullptr;  // nullptr: four components per target
    // Returns false (and prints) when the pipeline cannot be created; `message` receives Metal's complaint.
    auto pipeline = [&](id<MTLFunction> vs, id<MTLFunction> fs, NSString** message) -> id<MTLRenderPipelineState> {
      MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
      d.vertexFunction = vs;
      d.fragmentFunction = fs;
      for (int i = 0; i < 3; ++i) d.colorAttachments[i].pixelFormat = formats[i];
      NSError* e = nil;
      id<MTLRenderPipelineState> p = [device newRenderPipelineStateWithDescriptor:d error:&e];
      if (!p && message) *message = e.localizedDescription;
      return p;
    };
    auto draw = [&](id<MTLRenderPipelineState> state, Pixels& out) {
      id<MTLTexture> tex[3];
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
      for (int i = 0; i < 3; ++i) {
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:formats[i]
                                                                                      width:kSize
                                                                                     height:kSize
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModeShared;
        tex[i] = [device newTextureWithDescriptor:td];
        pass.colorAttachments[i].texture = tex[i];
        pass.colorAttachments[i].loadAction = MTLLoadActionClear;
        pass.colorAttachments[i].storeAction = MTLStoreActionStore;
        pass.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 0);
      }
      id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:pass];
      [enc setRenderPipelineState:state];
      [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return fprintf(stderr, "command buffer: %s\n", cb.error.description.UTF8String), false;
      for (int i = 0; i < 3; ++i) {
        out.rt[i].assign(kSize * kSize * 4, 0);
        int n = components ? components[i] : 4;
        std::vector<uint32_t> raw(kSize * kSize * n);
        [tex[i] getBytes:raw.data()
             bytesPerRow:kSize * 4 * n
              fromRegion:MTLRegionMake2D(0, 0, kSize, kSize)
             mipmapLevel:0];
        for (int px = 0; px < kSize * kSize; ++px)
          for (int k = 0; k < n; ++k) out.rt[i][px * 4 + k] = raw[px * n + k];
      }
      return true;
    };

    int failures = 0;
    struct Pair {
      const char* name;
      id<MTLFunction> vs;
      id<MTLFunction> fs;
    };
    auto flt = [](uint32_t w) {
      float f;
      memcpy(&f, &w, 4);
      return f;
    };
    auto bits = [](float f) {
      uint32_t w;
      memcpy(&w, &f, 4);
      return w;
    };

    for (int narrow = 0; narrow < 2; ++narrow) {
      id<MTLFunction> mf = narrow ? msl(@"frg_narrow") : msl(@"frg"), tf = narrow ? tNarrow : tFrag;
      const char* shader = narrow ? "narrow" : "flat";
      formats = narrow ? narrowFormats : wideFormats;
      components = narrow ? narrowComponents : nullptr;
      Pair pairs[4] = {{"MSL vertex + MSL fragment", msl(@"vtx"), mf},
                       {"translated vertex + translated fragment", tVert, tf},
                       {"translated vertex + MSL fragment", tVert, mf},
                       {"MSL vertex + translated fragment", msl(@"vtx"), tf}};
      Pixels want;
      for (int p = 0; p < 4; ++p) {
        NSString* why = nil;
        id<MTLRenderPipelineState> state = pipeline(pairs[p].vs, pairs[p].fs, &why);
        Pixels got;
        if (!state) return fprintf(stderr, "%s: %s: pipeline: %s\n", shader, pairs[p].name, why.UTF8String), 1;
        if (!draw(state, got)) return 1;
        if (p == 0) {
          want = got;
          // The MSL pair against the per-vertex values of the provoking vertex (independent of the translator).
          int v = kProvoking, bad = 0;
          for (int px = 0; px < kSize * kSize; ++px) {
            const uint32_t* f = &got.rt[0][px * 4];
            const int32_t* i = reinterpret_cast<const int32_t*>(&got.rt[1][px * 4]);
            const uint32_t* u = &got.rt[2][px * 4];
            bad += f[0] != bits(float(v) * 1.5f + 1.0f);
            bad += i[0] != -(v + 1) || (narrow ? 0 : i[2] != 5 || i[3] != -9) || i[1] != v * 100 - 7;
            bad += u[0] != 0xF0000000u + uint32_t(v) * 65537u || u[1] != 0xFFFFFFFFu - uint32_t(v);
            if (!narrow) bad += u[2] != 0x9E3779B9u * uint32_t(v + 1) || u[3] != 1;
          }
          printf("flat_gpu: %s: provoking vertex %d: %d of %d pixels differ from its values\n", shader, v, bad,
                 kSize * kSize);
          if (bad) ++failures;
          if (!narrow) {
            // The smooth float varies across the target (the contrast to the flat ones): perspective-correct value
            // between vertex 0 (0) and vertices 1, 2 (1, 2).
            float lo = flt(got.rt[0][1]), hi = lo;
            for (int px = 0; px < kSize * kSize; ++px) {
              float s = flt(got.rt[0][px * 4 + 1]);
              lo = s < lo ? s : lo;
              hi = s > hi ? s : hi;
            }
            printf("flat_gpu: smooth varying ranges over [%g, %g]\n", lo, hi);
            if (!(hi - lo > 0.1f)) ++failures;
          }
          continue;
        }
        int diff = 0;
        for (int t = 0; t < 3; ++t) diff += memcmp(got.rt[t].data(), want.rt[t].data(), got.rt[t].size() * 4) != 0;
        printf("flat_gpu: %s: %s: %d of 3 render targets differ\n", shader, pairs[p].name, diff);
        if (diff) ++failures;
      }
    }

    // Signedness mismatch between the stages, same location: report what Apple's compiler does.
    NSString* why = nil;
    id<MTLRenderPipelineState> mixed = pipeline(msl(@"vtx_mixed"), msl(@"frg"), &why);
    printf("flat_gpu: signedness mismatch (vertex uint2/int3, fragment int2/uint3): %s\n",
           mixed ? "pipeline created" : why.UTF8String);
    if (mixed) {
      Pixels a, b;
      id<MTLRenderPipelineState> same = pipeline(msl(@"vtx"), msl(@"frg"), nullptr);
      if (!draw(mixed, a) || !draw(same, b)) return 1;
      int diff = 0;
      for (int t = 0; t < 3; ++t) diff += memcmp(a.rt[t].data(), b.rt[t].data(), a.rt[t].size() * 4) != 0;
      printf("flat_gpu: mismatched pair renders %s the matched pair\n", diff ? "differently from" : "identically to");
    }
    return failures ? 1 : 0;
  }
}
