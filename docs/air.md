# AIR output notes

These notes describe the writer in this project. They are not an Apple specification or a promise that future drivers accept every emitted form.

AIR modules use a 64-bit little-endian target, triple `air64_v26-apple-macosx14.0.0`, AIR version 2.6 and Metal language version 3.1. `airbc` owns a small IR with typed pointers and writes LLVM-13-layout bitcode without linking LLVM. Address spaces are private 0, device 1, constant 2 and threadgroup 3. Bitcode is enclosed in a 20-byte wrapper with its offset and length; the wrapper magic is expressed in source as an integer constant.

Function metadata identifies an `air.vertex`, `air.fragment` or `air.kernel` entry. Argument metadata supplies the Metal location, access, address space and type/name descriptions. Builtins such as `air.vertex_id`, `air.position`, `air.sample_id` and `air.thread_position_in_grid` identify special inputs. Graphics returns an aggregate of vertex/fragment outputs; compute returns void. Tessellation evaluation is emitted as a vertex function with `air.patch`, domain and control-point count metadata.

The front end decodes types, constants, decorations and variables, translates blocks and phi nodes, then promotes local allocations, scalarizes elementwise vector expressions, eliminates redundant expressions, simplifies instructions/control flow and optionally threads dispatch jumps. Module construction order and writer options affect the byte stream. `kVersion` is a cache-format identity, not an AIR version.

Math, texture, barrier and atomic operations call `air.*` intrinsics. Approximate (`air.fast_*`) intrinsic calls carry no fast-math flags; compute module metadata selects fast math disabled while both graphics and compute request denormal flushing. Caller-visible numerical exceptions are documented in the README. `PackHalf2x16` is the IEEE binary16 conversion (round to nearest even, overflow to infinity).

`airkit` builds compute kernels using the same writer. Declare argument buffers, textures (`float`, `half`, `int` or `uint` texels), samplers or builtins before `Kernel::begin()`, use its builder and intrinsic helpers, terminate every block, and call `bitcode()`. `airkit::buildMetallib` packs the named kernels. SIMD and matrix helpers in airkit do not imply corresponding SPIR-V front-end support.

## Platforms

`Options::platform` selects macOS (default), iOS or iOS Simulator, and `minOsMajor`/`minOsMinor` the minimum OS version. 0 selects macOS 14 (the first with Metal 3.1) or iOS 18. iOS 18 is the floor for iOS and iOS Simulator (xenia's iOS deployment target) and older requests are raised to it. The generated code is identical on all platforms. Probing Apple's toolchain with the same Metal 3.1 source (compute, vertex, fragment and post-tessellation vertex functions) for macOS, iPhoneOS and iPhoneSimulator shows only these differences, all derived from the deployment target:

| | macOS 14 | iOS 18 | iOS Simulator 18 | iOS 26 |
|---|---|---|---|---|
| module triple | `air64_v26-apple-macosx14.0.0` | `air64_v27-apple-ios18.0.0` | `air64_v27-apple-ios18.0.0-simulator` | `air64_v28-apple-ios26.0.0` |
| metallib platform field (header offset 4) | `0x8001` | `0x0001` | `0x0001` | `0x0001` |
| OS byte (offset 11) | `0x81` | `0x82` | `0x87` | `0x82` |
| OS version (offset 12, major and minor u16) | 14.0 | 18.0 | 18.0 | 26.0 |
| AIR version (`air.version`, per-function `VERS`) | 2.6 | 2.7 | 2.7 | 2.8 |
| file version | 2.7 | 2.8 | 2.8 | 2.9 |

The AIR version follows the minimum OS: macOS 2.6 for 14, 2.7 for 15, 2.8 from 26; iOS 2.7 for 18, 2.8 from 26; the file version minor is the AIR minor plus one. The triple suffix `air64_vNN` mirrors the AIR version (`v28` for AIR 2.8). Both platforms copy Apple's spelling. Versions above 26 are written as 26 until a newer toolchain is targeted. Data layout, module flags (`air.max_*`), the Metal language version (3.1), function tags, file type (0), and the TESS tag are the same on every platform.

The Metal front end accepts every intrinsic this compiler emits (simd shuffles and ballots, simdgroup matrices, atomics, cube-array and 3D sampling, read-write textures, framebuffer fetch, demote, clip distances, sample masks, base vertex, quad and triangle post-tessellation vertex functions) for iOS, and the IR is identical to macOS. GPU-family limits (for example simdgroup matrices needing A14 or later, or the 64-bit resource handles of the bindless heaps needing argument buffers tier 2) are enforced by the driver at pipeline creation, not by the offline tools, so they need a device check. No construct is rejected by platform.

## Function constants

Observed by compiling small MSL sources of our own with Apple's public compiler (`xcrun metal`, macOS 14 and iOS 18 targets; the output is identical apart from the triple and AIR version) and reading them back with `llvm-dis` and `metal-objdump`. `constant uint4 fc3 [[function_constant(3)]];` compiles to:

- one internal global holding the value, `@_ZL3fc3 = internal unnamed_addr addrspace(2) global <4 x i32> undef, align 16`, and one initializer, `@_Z3fc3.MTL_FC_INIT_3_Dv4_j = internal unnamed_addr addrspace(2) externally_initialized constant <4 x i32> undef, section "air.fc_initializer", align 16` (the name is the Itanium-style `_Z<length><name>.MTL_FC_INIT_<index>_<type code>`; `Dv4_j` is uint4, `j` uint, `b` bool). Metal replaces the initializer at specialization (`newFunctionWithName:constantValues:`);
- a static initializer (`llvm.global_ctors`, a function in section `air.static_init`) that copies each initializer into its global, and shader code that loads the global;
- named metadata `!air.function_constants`, one node per constant the function uses: `!{ptr addrspace(2) @_Z3fc3.MTL_FC_INIT_3_Dv4_j, !"uint4", !"fc3", i32 3}`. In the linked per-function module of a metallib the node gains a fifth operand `i1 true` (required) and the globals are internal. The type name is the MSL spelling (`uint4`, `uint`, `bool`); a bool global is `i8` with `!range`;
- an unused constant leaves nothing behind (no global, no metadata, no CNST entry).

In the metallib function record, the public metadata `CNST` tag has `u16 count` and, per constant, `name\0`, a `u8` MTLDataType (uint 33, uint4 36, bool 53), a `u16` function constant index and a `u8` required flag (1 for every constant a function uses). Only the constants a function uses are listed, in index order. Both Metal and the validator accept the same records on macOS and iOS. Metal exposes them in `MTLFunction.functionConstantsDictionary` (name, type, index, required).

spirv2air (`GraphicsBindings::functionConstantSet`) writes the same module without the static initializer: it loads straight from the `air.fc_initializer` constant, which Metal specializes (checked on the GPU in `tests/fc_gpu.mm`), and drops globals that are not read after optimization. Metal does not reject a specialization that omits a required constant: like the output of Apple's compiler, the pipeline is built and the constant is undefined, while the unspecialized function from `newFunctionWithName:` cannot be used in a pipeline.
