// Jump threading of the Xenos dispatch loop on the GPU (src/spirv2air/jumpthread.cpp). The shader in
// tests/data/dispatch/dispatch.spvasm is a hand-written dispatch loop: forward jumps, conditional jumps to constants
// (by branches and by OpSelect), a bounded backward jump and one data-dependent pc. It is translated with the pass and
// without it (Options::jumpThread), both run, and the outputs must equal each other and a CPU model of the program.
// usage: dispatch_gpu <dispatch.spv>
#import <Metal/Metal.h>

#include <spirv2air.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

constexpr unsigned kThreads = 128;

std::vector<uint32_t> readWords(const char* path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<uint32_t> words(f ? size_t(f.tellg()) / 4 : 0);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
  return words;
}

std::vector<uint32_t> model(const std::vector<uint32_t>& in) {
  std::vector<uint32_t> out(kThreads * 2, 0xDEADBEEFu);
  for (uint32_t g = 0; g < kThreads; ++g) {
    uint32_t pc = 0, acc = g ^ in[g & 63], cnt = 0;
    for (bool run = true; run;) switch (pc) {
        case 0:
          acc = acc * 3 + 1;
          pc = (g & 1) ? 2 : 1;
          break;
        case 1:
          acc += 10;
          pc = 3;
          break;
        case 2:
          acc ^= 0x55;
          pc = (acc & 4) ? 3 : 1;
          break;
        case 3:
          ++cnt;
          acc += cnt;
          pc = cnt < (g & 3) + 2 ? 2 : (g == 7 ? in[64] : 4);
          break;
        case 4:
          out[g] = acc;
          out[128 + g] = cnt;
          [[fallthrough]];
        default:
          run = false;
      }
  }
  return out;
}

bool run(id<MTLDevice> device, id<MTLCommandQueue> queue, const std::vector<uint32_t>& words,
         const std::vector<uint32_t>& in, bool thread, std::vector<uint32_t>& out, size_t& libSize) {
  spirv2air::Options options;
  options.jumpThread = thread;
  options.bindings = {{0, 0, 0, 0}, {0, 1, 1, 0}};
  spirv2air::Result r = spirv2air::translate(words.data(), words.size(), options);
  if (!r.ok()) return fprintf(stderr, "translate failed: %s\n", r.error.c_str()), false;
  libSize = r.metallib.size();
  NSError* error = nil;
  dispatch_data_t data = dispatch_data_create(r.metallib.data(), r.metallib.size(), nullptr, nullptr);
  id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
  if (!library) return fprintf(stderr, "newLibraryWithData: %s\n", error.description.UTF8String), false;
  id<MTLFunction> function = [library newFunctionWithName:@"main"];
  id<MTLComputePipelineState> pso = function ? [device newComputePipelineStateWithFunction:function error:&error] : nil;
  if (!pso) return fprintf(stderr, "pipeline: %s\n", error.description.UTF8String), false;
  std::vector<uint32_t> init(kThreads * 2, 0xDEADBEEFu);
  id<MTLBuffer> bin = [device newBufferWithBytes:in.data() length:in.size() * 4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bout = [device newBufferWithBytes:init.data()
                                           length:init.size() * 4
                                          options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [queue commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:pso];
  [e setBuffer:bin offset:0 atIndex:0];
  [e setBuffer:bout offset:0 atIndex:1];
  [e dispatchThreadgroups:MTLSizeMake(kThreads / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
  [e endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
  if (cb.status != MTLCommandBufferStatusCompleted)
    return fprintf(stderr, "command buffer failed: %s\n", cb.error.description.UTF8String), false;
  out.assign(static_cast<const uint32_t*>(bout.contents), static_cast<const uint32_t*>(bout.contents) + init.size());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return fprintf(stderr, "usage: dispatch_gpu dispatch.spv\n"), 2;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return fprintf(stderr, "no Metal device\n"), 2;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    std::vector<uint32_t> words = readWords(argv[1]);
    std::vector<uint32_t> in(65);
    for (unsigned i = 0; i < 64; ++i) in[i] = 0x9E3779B9u * (i + 1) >> (i % 7);
    in[64] = 4;
    std::vector<uint32_t> want = model(in), threaded, plain;
    size_t sizeThreaded = 0, sizePlain = 0;
    if (!run(device, queue, words, in, true, threaded, sizeThreaded) ||
        !run(device, queue, words, in, false, plain, sizePlain))
      return 1;
    unsigned failures = 0;
    for (unsigned i = 0; i < want.size(); ++i) {
      if (threaded[i] != want[i] && ++failures <= 10)
        fprintf(stderr, "FAIL threaded [%u]: 0x%08x, expected 0x%08x\n", i, threaded[i], want[i]);
      if (plain[i] != want[i] && ++failures <= 10)
        fprintf(stderr, "FAIL plain [%u]: 0x%08x, expected 0x%08x\n", i, plain[i], want[i]);
    }
    if (sizeThreaded == sizePlain) fprintf(stderr, "note: the pass did not change the module size\n");
    if (failures) return fprintf(stderr, "dispatch_gpu: %u value failures\n", failures), 1;
    printf("dispatch_gpu: threaded and plain dispatch loops match each other and the model\n");
  }
  return 0;
}
