// simplifyInstructions' constant folding of AIR intrinsic calls and int -> float casts, checked against Apple's
// compiler and GPU: for each case of tests/fold_cases.h, a kernel reads every tuple's operands from a buffer and
// computes the call on the GPU, and a second kernel holds the same calls on constants, folded by
// simplifyInstructions before it is built. Every tuple that folds must give the GPU's bits; tuples that must not fold
// are only checked to stay calls (their results are undefined or depend on the rounding). Runs with fast math on and
// off (the module's compile option).
// usage: fold_gpu [case-name-substring]
#import <Metal/Metal.h>

#include <cstdio>
#include <cstring>

#include "fold_cases.h"

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

void run(id<MTLCommandQueue> q, id<MTLComputePipelineState> p, NSArray<id<MTLBuffer>>* bufs, size_t threads) {
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:p];
  for (NSUInteger i = 0; i < bufs.count; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i];
  [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(threads, 64), 1, 1)];
  [enc endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

}  // namespace

int main(int argc, char** argv) {
  @autoreleasepool {
    const char* filter = argc > 1 ? argv[1] : "";
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) { std::printf("no Metal device\n"); return 77; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    int failures = 0;
    size_t compared = 0;
    for (const foldcases::Case& c : foldcases::cases()) {
      if (!std::strstr(c.name.c_str(), filter)) continue;
      for (int fast = 1; fast >= 0; --fast) {
        std::string label = c.name + (fast ? " (fast)" : " (strict)");
        airkit::Options o;
        o.fastMath = fast;
        airkit::Kernel kc("k", o), kr("k", o);
        auto stores = foldcases::buildConstant(kc, c);
        int bad = foldcases::checkFolds(kc, c, stores);
        foldcases::buildRuntime(kr, c);
        std::string err;
        id<MTLComputePipelineState> pc = pipeline(dev, kc, err), pr = pc ? pipeline(dev, kr, err) : nil;
        if (!pc || !pr) {
          std::printf("FAIL %s: %s\n", label.c_str(), err.c_str());
          ++failures;
          continue;
        }
        const size_t n = c.tuples.size();
        std::vector<uint32_t> in;
        for (const auto& t : c.tuples) in.insert(in.end(), t.bits.begin(), t.bits.end());
        id<MTLBuffer> bin = [dev newBufferWithBytes:in.data() length:in.size() * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outc = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> outr = [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared];
        run(q, pc, @[ outc ], 1);
        run(q, pr, @[ bin, outr ], n);
        const uint32_t* a = static_cast<const uint32_t*>(outc.contents);
        const uint32_t* g = static_cast<const uint32_t*>(outr.contents);
        for (size_t t = 0; t < n; ++t) {
          if (!c.tuples[t].fold) continue;
          ++compared;
          if (a[t] != g[t] && bad++ < 4) {
            std::printf("  %s(", c.name.c_str());
            for (size_t k = 0; k < c.tuples[t].bits.size(); ++k) std::printf("%s0x%08x", k ? ", " : "", c.tuples[t].bits[k]);
            std::printf("): folded 0x%08x, GPU 0x%08x\n", a[t], g[t]);
          }
        }
        std::printf("%s %s\n", bad ? "FAIL" : "ok  ", label.c_str());
        failures += bad != 0;
      }
    }
    std::printf("%zu folded results compared with the GPU\n", compared);
    std::printf(failures ? "%d failing\n" : "all passed\n", failures);
    return failures ? 1 : 0;
  }
}
