# spirv2air

spirv2air compiles a subset of SPIR-V straight to Apple AIR and writes a loadable metallib. It is BSD-3-Clause C++17, needs no LLVM, and runs no shader compiler at runtime. It includes its own LLVM bitcode writer (`airbc`), metallib container writer (`mtlb`), and compute-kernel builder (`airkit`).

It started as the shader backend for the xenia emulator's Metal renderer. The default graphics layout follows that integration; `spirv2air::Options` configures it. It is not a general Vulkan implementation.

## Why

Metal accepts shaders in two forms: Metal Shading Language source, which the driver compiles at runtime, or a metallib that already holds AIR, the LLVM-based intermediate representation Metal's GPU compilers consume.

Most translation layers take the source route. MoltenVK converts SPIR-V to MSL with SPIRV-Cross and compiles it with `newLibraryWithSource:`. Mesa's KosmicKrisp converts NIR to MSL and compiles it through the Metal 4 compiler (`MTL4LibraryDescriptor.source`). Either way, every new shader is printed as MSL and then parsed, type-checked and lowered to AIR again at runtime before the GPU backend sees it.

The other route is DXIL to AIR through Apple's Metal Shader Converter. That converter is a closed-source binary library, so a project using it must ship Apple's library and cannot change or debug how it lowers code.

spirv2air takes the AIR route without the closed dependency:
- It lowers SPIR-V straight to AIR bitcode, writes the metallib container itself, and loads it with `newLibraryWithData:`.
- There is no MSL source step, no LLVM, and no Apple converter library at build or run time.
- Output is deterministic, so it can be cached by content.

The driver still compiles AIR to GPU machine code at pipeline creation. spirv2air removes the source-to-AIR stage, not that final step.

## Targets

| Platform | Minimum OS | AIR | Triple |
| --- | --- | --- | --- |
| macOS (default) | 14.0 | 2.6 | `air64_v26-apple-macosx14.0.0` |
| iOS | 18.0 | 2.7 | `air64_v27-apple-ios18.0.0` |
| iOS Simulator | 18.0 | 2.7 | `air64_v27-apple-ios18.0.0-simulator` |

A higher minimum OS selects a newer AIR version (for example macOS 26 or iOS 26 gives AIR 2.8); see [Platforms](docs/air.md#platforms). Select a target with `Options::platform` and `minOsMajor`/`minOsMinor`, or with the CLI flags `--platform` and `--min-os`.

The GPU tests run on Apple-silicon Macs. iOS output has run on an iPhone through xenia. Building requires the macOS SDK; output is little-endian only.

## Build and test

You need the Xcode command-line tools, CMake 3.20+, Ninja, SPIRV-Headers, and SPIRV-Tools (`spirv-as`, for the test fixtures). CMake does not download dependencies.

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build -L CPU --output-on-failure
ctest --test-dir build -L GPU --output-on-failure -j 1   # needs a Metal device
```

| CMake option | Default | Effect |
| --- | --- | --- |
| `SPIRV2AIR_SPIRV_HEADERS_DIR` | searched | SPIRV-Headers include directory |
| `SPIRV2AIR_BUILD_TRANSLATOR` | `ON` | `OFF` builds only `airbc`, `mtlb` and `airkit`; no SPIRV-Headers needed, and no CLI or tests |
| `SPIRV2AIR_BUILD_CLI` | top level | Build `spirv2air_cli` |
| `SPIRV2AIR_BUILD_TESTS` | top level | Build the tests (also gated by `BUILD_TESTING`) |
| `SPIRV2AIR_BUILD_GPU_TESTS` | `ON` | Build the tests that need a Metal device |
| `SPIRV2AIR_ENABLE_VERIFY` | `OFF` | IR verifiers in optimized builds (always on in Debug) |

CPU tests cover translation, configurable bindings, rejection diagnostics, determinism, the CLI, and container round trips. GPU tests compare results against hand-written MSL on the device.

## Usage

### C++ API

```cpp
#include <spirv2air.h>

// words holds a validated SPIR-V module.
spirv2air::Options options;
options.bindings = {{0, 0, 0, 0}};  // set, binding, Metal buffer index, Metal texture index
// options.platform = spirv2air::Platform::IOS;  // iOS 18+
spirv2air::Result result = spirv2air::translate(words.data(), words.size(), options);
if (!result.ok()) {
  // result.error names the rejected construct.
} else {
  // result.metallib holds the library; load it with -[MTLDevice newLibraryWithData:error:].
}
```

- **Input must already be validated** (for example with `spirv-val`). spirv2air is not a defensive validator for untrusted SPIR-V.
- **Cache keys** must include the input module, every option, and `spirv2air::kVersion`.
- **Graphics shaders** use the layout in [the binding contract](docs/bindings.md): resource and sampler heaps, a top-level argument buffer, texture index records, and the xenia-derived defaults (`vertexIndexRelativeToBase`, `positionInfToNan`) that can be turned off.

### CMake

| Use | How |
| --- | --- |
| Source dependency | `add_subdirectory(path/to/spirv2air)` and link `spirv2air::spirv2air`. Tests and the CLI are off when embedded. |
| AIR libraries only | Also set `SPIRV2AIR_BUILD_TRANSLATOR=OFF` and link `spirv2air::airkit` (or `airbc`, `mtlb`). |
| Installed package | `cmake --install build --prefix PREFIX`, then `find_package(spirv2air 1.0 CONFIG REQUIRED)`. |
| Direct include | `include(path/to/spirv2air/cmake/spirv2air.cmake)` defines the library targets only. It may be included more than once. |

### Command line

```sh
spirv-as --target-env vulkan1.1 tests/data/compute/ids.spvasm -o build/ids.spv
build/spirv2air_cli build/ids.spv build/ids.metallib --buffer 0:0:0
```

All flags and the batch manifest format are in [docs/cli.md](docs/cli.md).

## Supported SPIR-V

A module has one entry point and one function with no parameters. Supported execution models:
- `Vertex`;
- `Fragment`;
- `TessellationEvaluation`, for triangle and quad domains, emitted as a Metal post-tessellation vertex function;
- `GLCompute`, with a literal `LocalSize`.

Supported capabilities: `Shader`, `DenormFlushToZero`, `DerivativeControl`, `DemoteToHelperInvocation`, `ImageQuery`, `Tessellation`, `SampleRateShading`, `ClipDistance`, `CullDistance` and `FloatControls2` (strict `FPFastMathMode None` only; other masks and `FPFastMathDefault` are rejected). A supported capability does not mean every operation it allows is implemented.

**Types:** booleans, 32-bit integers and floats, vectors, arrays and structs. Buffers need explicit member offsets and array strides.

**Compute:**
- storage and uniform buffers, push constants;
- private and workgroup variables, invocation and workgroup IDs;
- workgroup and subgroup control barriers;
- relaxed 32-bit integer atomics, including load, store, increment/decrement and compare-exchange;
- texel fetches from 2D textures: multisampled, integer and float.

**Graphics:**
- float, 32-bit int and uint scalar and vector varyings by location (integer fragment inputs must be `Flat`, as Vulkan requires), `Flat` float varyings (flat values come from the first vertex of a triangle, as in Vulkan's default);
- depth and sample-mask outputs; sample ID and sample-mask inputs;
- centroid, sample, perspective and no-perspective interpolation;
- discard and demote-to-helper; derivatives;
- bindless float texture sampling of 2D, 2D-array, 3D, cube and cube-array textures, with `Bias`, `Lod` and `Grad` operands;
- up to eight combined clip and cull distances. Cull distances are approximated with a steep clip distance; see the comment in `translate.cpp`;
- storage buffers at configured argument-buffer offsets, supporting loads and 32-bit integer atomics;
- uint4 uniforms mapped to Metal function constants.

**Math:**
- `GLSL.std.450`:
  - rounding: `Floor`, `Ceil`, `Trunc`, `Round`, `RoundEven`, `Fract`;
  - float: `FAbs`, `FSign`, `Sqrt`, `InverseSqrt`, `Exp`, `Exp2`, `Log`, `Log2`, `Pow`, `Sin`, `Cos`, `Tan`, `Asin`, `Acos`, `Atan`, `Atan2`, `Sinh`, `Cosh`, `Tanh`, `Asinh`, `Acosh`, `Atanh`, `Radians`, `Degrees`, `Fma`, `Ldexp`, `Modf`, `ModfStruct`, `Frexp`, `FrexpStruct`;
  - interpolation: `FMix`, `Step`, `SmoothStep`;
  - geometric: `Length`, `Distance`, `Cross` (3-component), `Normalize`, `FaceForward`, `Reflect`, `Refract`;
  - min, max and clamp: `FMin`, `FMax`, `FClamp`, `NMin`, `NMax`, `NClamp`, `SMin`, `SMax`, `SClamp`, `UMin`, `UMax`, `UClamp`;
  - integer: `SAbs`, `SSign`, `FindILsb`, `FindSMsb`, `FindUMsb`;
  - packing: `PackHalf2x16`, `UnpackHalf2x16`, and Pack/Unpack `Snorm4x8`, `Unorm4x8`, `Snorm2x16`, `Unorm2x16`.
- `Fract` is `x - floor(x)`.
- The new trigonometric, interpolation and geometric operations follow Apple's compiler for the same MSL functions. `Step` follows the SPIR-V wording (0 if `x < edge`, otherwise 1, so a NaN operand gives 1; MSL `step` gives 0), `FMix` is MSL `mix`, and in compute `Length`/`Normalize` keep MSL's overflow and underflow rescaling.
- `PackHalf2x16` rounds to nearest even, overflows to infinity, and keeps NaN.
- Zero-count bitfield extract gives 0; zero-count insert keeps the base.
- Graphics math uses Metal's fast (`air.fast_*`) intrinsics. Denormals are flushed.

[docs/spirv-op-coverage.md](docs/spirv-op-coverage.md) has the per-family coverage table and the full list of implemented opcodes.

## Not supported

Anything outside the subset above is rejected with a diagnostic. Some unrelated module-level records may be ignored. Translate only validated SPIR-V, and check a new family of shaders on the GPU.

| Area | Not supported | Why |
| --- | --- | --- |
| Execution models | Tessellation control, mesh/task, ray tracing | Not implemented yet |
| Execution models | Geometry; isoline tessellation | Metal has no geometry or isoline stage; these would need emulation |
| Program structure | Function calls and parameters, multiple entry points | Not implemented yet |
| Types | Matrices; 8-, 16- and 64-bit numeric types | Not implemented yet |
| Constants | Specialization constants | Not implemented yet (function constants cover the xenia case) |
| Math | `GLSL.std.450` operations not listed above: `Determinant`, `MatrixInverse` (no matrix types), `IMix`, `InterpolateAt*`, `PackDouble2x32`, `UnpackDouble2x32`; other extended instruction sets | Not implemented yet |
| Varyings | Arrays, matrices, structs or blocks as varyings; 8-, 16- and 64-bit varyings | Not implemented yet |
| Resources | Descriptor arrays and runtime descriptor indexing; graphics push constants | Not implemented yet |
| Images | Storage images and image writes; graphics `OpImageFetch`; sampling from compute shaders | Not implemented yet |
| Sampling | Depth-compare, gather, projective and sparse sampling; `MinLod` and `ConstOffset` operands | Not implemented yet |
| Atomics and memory | Non-relaxed memory order; standalone memory barriers | Need a Metal 3.2 (newer OS) target |
| Subgroups | Subgroup arithmetic and ballot instructions | Not implemented yet |

## Libraries and documentation

- **Headers:**
  - `src/airkit/airkit.h`: compute kernel arguments (buffers, textures with float, half, int or uint texels, samplers, builtins), AIR intrinsic helpers, and library packing.
  - `src/airbc/airbc.h`: the IR and bitcode writer.
  - `src/metallib/metallib.h`: container sizing and writing.
- **Docs:**
  - [AIR notes](docs/air.md) and [container notes](docs/metallib.md) describe this implementation's output.
  - [The binding contract](docs/bindings.md) describes the graphics layout.
  - [CLI](docs/cli.md) covers the command-line tool.
- [CONTRIBUTING.md](CONTRIBUTING.md) covers the test policy, the verifier build and the debugging switches.
- Licence: [LICENSE](LICENSE).
