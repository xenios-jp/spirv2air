# Contributing

Use hand-written SPIR-V assembly under `tests/data/` for translator fixtures. Keep tests small and self-contained, with an explanation of the behavior and expected result. Add new assembly inputs to the CMake assembly list; keep generated SPIR-V and shader libraries in the build directory. Runtime reference kernels in GPU tests are our own hand-written Metal source. Do not add precompiled libraries, extracted shaders, application assets, or external reference outputs.

Before submitting a compiler change, run the CPU tests and the relevant GPU tests. CPU tests exercise translation, diagnostics, deterministic output, configurable layouts, CLI modes and synthetic container round trips. They cannot establish driver acceptance or numerical correctness on a GPU. Run the GPU tests with a Metal device, record the OS and device, and compare against the CPU model or hand-written Metal kernel used by the test. Run them with `ctest --test-dir build -L GPU --output-on-failure -j 1`.

Enable the IR structure and dominance verifiers even with optimized host code:

```sh
cmake -S . -B build-verify -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPIRV2AIR_ENABLE_VERIFY=ON -DSPIRV2AIR_BUILD_GPU_TESTS=OFF
cmake --build build-verify
ctest --test-dir build-verify -L CPU --output-on-failure
```

Debug builds also enable these verifiers. Invalid IR returns a translation diagnostic before serialization. Tests should cover new behavior, rejected unsupported forms and any affected option defaults. Compare output bytes against the preceding version for unchanged inputs/options. `kVersion` is an output cache identity: a change that alters existing default output requires a version decision; it is not changed merely for build or documentation cleanup. Cache keys must include every option.

The implementation targets the execution models and restricted instruction set documented in the README. Document operand, type, memory-order and stage restrictions with new lowering. Use the SPIR-V and Metal language semantics to explain behavior; clearly identify approximations and legacy host conventions. Format notes describe our implementation and should not imply a vendor specification or a general conformance guarantee.

## Debugging

- `S2A_DUMP_DIR=<dir>` writes numbered `.bc` files after each pass, plus the final metadata, for `llvm-dis` and similar tools. Create the directory first and use one per translation, because file names repeat. Dumping verifies the intermediate IR and does not change the output bytes.
- `S2A_STAGE_TIMES` enables process-wide timing counters. It is meant for single-threaded profiling.

## Amber conformance runner

`tests/amber/amber_runner.mm` (built with `-DSPIRV2AIR_BUILD_AMBER_RUNNER=ON`) runs the compute subset of the Khronos VK-GL-CTS Amber scripts through spirv2air on a Metal device and classifies each script as PASS, FAIL, TRANSLATE_REJECT, UNSUPPORTED_AMBER, CRASH or HANG. It needs `glslangValidator` and `spirv-as` (and `spirv-opt` for `--inline`). The CTS data is external and is never committed here; pass the directory `external/vulkancts/data/vulkan/amber` of a VK-GL-CTS checkout with `--cts`. `tests/amber/run_batches.sh` runs everything in lock-protected batches and writes a CSV.
