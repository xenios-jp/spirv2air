# Shader operation coverage

Coverage audit for the 32-bit Shader profile on the macOS 14 / AIR 2.6 / Metal 3.1 target. Scalar and vector
operands are supported unless a row says otherwise.
This is an audit of common operations, not a claim that every SPIR-V capability or execution model is implemented. Unsupported instructions return a translation error.

| Family | Operations | Status |
| --- | --- | --- |
| Signed division | `OpSDiv`, `OpSRem`, `OpSMod` | Supported, including guarded zero divisors and signed overflow |
| Unsigned division | `OpUDiv`, `OpUMod` | Supported; zero divisors are guarded |
| Floating remainder | `OpFRem`, `OpFMod` | Supported; `OpFRem` uses the truncating AIR fmod, `OpFMod` the SPIR-V floor formula |
| Dynamic vectors | `OpVectorExtractDynamic`, `OpVectorInsertDynamic` | Supported, with a defined out-of-range choice |
| Extended integers | `OpIAddCarry`, `OpISubBorrow`, `OpUMulExtended`, `OpSMulExtended` | Supported for both structure members and every vector lane; native AIR signed/unsigned high products |
| Bit operations | `OpBitReverse`; `OpBitCount`, shifts, bitfield insert/extract | Supported, including zero-count extract/insert |
| Classification | `OpIsInf`; `OpIsNan` | Supported |
| Selection/copy | Vector/scalar `OpSelect`, `OpCopyObject`; `OpCopyLogical` | Supported |
| Quantization | `OpQuantizeToF16` | Supported: half rounding, then half subnormals flush to signed zero |
| Integer GLSL | `SAbs`, `SSign`, `UClamp`; signed/unsigned min/max and `SClamp` | Supported |
| Float GLSL | `FSign`, `Fma`, `Exp`, `Log`, `Pow` | Supported; Fma always uses a fused intrinsic |
| Normalized packing | Pack/Unpack Snorm/Unorm 4x8 and 2x16 | All eight, using AIR's pack/unpack intrinsics |
| Half packing | `PackHalf2x16`, `UnpackHalf2x16` | Supported; IEEE binary16 conversion |
| Decomposition | `Modf`, `ModfStruct`, `Frexp`, `FrexpStruct` | Supported, including signed zero and subnormal handling |
| Exponents/scans | `Ldexp`, `FindILsb`, `FindSMsb`, `FindUMsb` | Supported |
| Image queries | `OpImageQuerySize`, `OpImageQuerySizeLod`, `OpImageQueryLevels`, `OpImageQuerySamples` | Supported for the image types below |
| Image extraction/LOD | `OpImage`; `OpImageQueryLod` | Supported; LOD query on 2D sampled images |
| Atomic RMW | Add/sub, and/or/xor, signed/unsigned min/max, exchange | 32-bit operations; unsupported ordering/scope is rejected |
| Additional atomics | Load/store, integer increment/decrement, strong compare/exchange | Supported on device buffers and compute Workgroup memory; relaxed ordering |
| Control barriers | Workgroup/subgroup execution, uniform/workgroup/cross-workgroup/image memory flags | Supported in compute, including combined flags |
| Memory barriers | `OpMemoryBarrier`, non-relaxed atomic ordering | Explicitly rejected: standalone fences need a newer target |
| Other image/storage operations | Storage image read/write, sparse images, query format/order, other dimensions | Outside the image profile; rejected |
| Other atomic widths/forms | 64-bit/float/flag atomics, Kernel-only weak compare/exchange | Outside the Shader/32-bit capability profile; rejected |
| Trig, interpolation and geometric GLSL | `Round`, `Tan`, `Asin`, `Acos`, `Atan`, `Atan2`, `Sinh`, `Cosh`, `Tanh`, `Asinh`, `Acosh`, `Atanh`, `Radians`, `Degrees`, `FMix`, `Step`, `SmoothStep`, `Length`, `Distance`, `Cross`, `Normalize`, `FaceForward`, `Reflect`, `Refract` | Supported; checked bit for bit against MSL in compute (fast math off) and, with a small tolerance for `Refract`, in fragment |
| Other GLSL math | `Determinant`, `MatrixInverse`, `IMix`, `InterpolateAt*`, `PackDouble2x32`, `UnpackDouble2x32` | Not implemented |
| Broader type/function profile | Matrices, 16/64-bit SPIR-V types, multiple functions, other execution models/capabilities | Outside this compiler's profile |

Compute images are sampled, non-arrayed 2D textures (float, signed/unsigned integer, optionally multisampled).
Graphics images are float 2D arrays, non-arrayed cubes and 3D textures. Array size queries return the logical
layer count. Query format/order and weak compare/exchange are Kernel-profile operations, not gaps in Shader support.

Integer addition/subtraction/multiplication and negation carry no LLVM overflow flags: signed overflow wraps.
Shift counts are masked with 31. Integer division guards its operands before emitting the divide, including
`INT_MIN/-1`; selecting a fallback after an invalid LLVM divide would be insufficient. Zero divisors produce an
all-ones quotient (unsigned `UINT_MAX`, signed `-1`) and the dividend as the remainder/modulus. `INT_MIN/-1` produces
`INT_MIN` and remainder/modulus zero. These deterministic choices apply where SPIR-V leaves the result undefined.
Out-of-range dynamic extraction returns zero; insertion leaves the vector unchanged. Invalid indices never reach
LLVM's extract/insert instructions. `SAbs(INT_MIN)` wraps to `INT_MIN`. `FSign(NaN)` returns positive zero.
`Frexp` returns the original zero/NaN/infinity with exponent zero for those special inputs. `Modf` preserves the
input sign on a zero fraction, including infinite inputs. Normalized packing maps NaN to zero; rounding follows the native AIR builtin.

`coverage_intrinsics.metal` is our own MSL source for reproducing AIR names/signatures. Compile with
`xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fno-fast-math -S -emit-llvm`.
The minimum OS flag matters: current SDKs otherwise emit newer atomic intrinsic signatures. Metal 3.1 rejects `atomic_thread_fence`; Metal 3.2 exposes a sequentially consistent fence, so adding
standalone barriers or stronger atomics requires an explicit newer-target design. A control barrier is not a safe
replacement for a memory-only barrier, which may occur in divergent code.

Validation is performed in an explicit `CMAKE_BUILD_TYPE=Debug` build (no NDEBUG), so both IR structure and dominance verifiers run. New/expanded GPU fixtures are `integer` (SPIR-V 1.5,
for logical copy), `float_math`, `packing`, `atomic_extra`, and `fetch`. They validate with spirv-val, translate with
the internal debug verifier, and their emitted AIR passes Apple's LLVM verifier. The CPU references in
`compute_gpu.mm` cover wrapping/overflow, zero divisors, masked shifts, dynamic indexing, both extended-result
members, scalarized vector lanes, NaN/infinity/signed zero/subnormals, normalized and half packing, decomposition,
atomic old/new values, and texture dimensions/levels/samples. Exp/log/pow and unpacked normalized values use a
small floating-point tolerance; exact operations use bit comparisons, accepting any NaN payload. Arithmetic
references also accept the CPU-computed flush-to-zero alternative advertised by AIR; bitwise Frexp and Modf
fraction handling preserve subnormals exactly. Pow comparisons cover its defined GLSL domain.

## Execution and type profile

Execution models are Vertex, Fragment, TessellationEvaluation (triangle/quad post-tessellation vertex functions), and GLCompute. There is one entry point and one parameterless SPIR-V function. Internal helper functions may be emitted for strong compare/exchange. Shader scalar data are booleans and 32-bit integer/float values, with vectors, arrays, runtime buffer arrays and structs; general matrices and SPIR-V 8/16/64-bit numeric data remain unsupported. Compute resources use explicit Options bindings; graphics resources use the argument-buffer layout in docs/bindings.md.

Accepted capabilities are Shader, DenormFlushToZero, DerivativeControl, DemoteToHelperInvocation, ImageQuery, Tessellation, SampleRateShading, ClipDistance and CullDistance. Acceptance of a capability does not implement every associated operation. Extended instructions are GLSL.std.450 only. The integer/logical-copy fixture uses SPIR-V 1.5; the remaining new fixtures use 1.3.

## Complete implemented instruction inventory

This inventory covers body-lowering cases in this compiler, subject to the type, operand, stage, image, scope and ordering restrictions above. Module declarations, annotations and execution modes are decoded separately. OpMemoryBarrier has a diagnostic case and is deliberately excluded from the supported list. Every instruction not listed is outside the implemented body profile.

132 core instructions:

`OpAccessChain`, `OpAll`, `OpAny`, `OpAtomicAnd`, `OpAtomicCompareExchange`, `OpAtomicExchange`, `OpAtomicIAdd`, `OpAtomicIDecrement`, `OpAtomicIIncrement`, `OpAtomicISub`, `OpAtomicLoad`, `OpAtomicOr`, `OpAtomicSMax`, `OpAtomicSMin`, `OpAtomicStore`, `OpAtomicUMax`, `OpAtomicUMin`, `OpAtomicXor`, `OpBitCount`, `OpBitFieldInsert`, `OpBitFieldSExtract`, `OpBitFieldUExtract`, `OpBitReverse`, `OpBitcast`, `OpBitwiseAnd`, `OpBitwiseOr`, `OpBitwiseXor`, `OpBranch`, `OpBranchConditional`, `OpCompositeConstruct`, `OpCompositeExtract`, `OpCompositeInsert`, `OpControlBarrier`, `OpConvertFToS`, `OpConvertFToU`, `OpConvertSToF`, `OpConvertUToF`, `OpCopyLogical`, `OpCopyObject`, `OpDPdx`, `OpDPdxCoarse`, `OpDPdxFine`, `OpDPdy`, `OpDPdyCoarse`, `OpDPdyFine`, `OpDemoteToHelperInvocation`, `OpDot`, `OpExtInst`, `OpFAdd`, `OpFDiv`, `OpFMod`, `OpFMul`, `OpFNegate`, `OpFOrdEqual`, `OpFOrdGreaterThan`, `OpFOrdGreaterThanEqual`, `OpFOrdLessThan`, `OpFOrdLessThanEqual`, `OpFOrdNotEqual`, `OpFRem`, `OpFSub`, `OpFUnordEqual`, `OpFUnordGreaterThan`, `OpFUnordGreaterThanEqual`, `OpFUnordLessThan`, `OpFUnordLessThanEqual`, `OpFUnordNotEqual`, `OpIAdd`, `OpIAddCarry`, `OpIEqual`, `OpIMul`, `OpINotEqual`, `OpISub`, `OpISubBorrow`, `OpImage`, `OpImageFetch`, `OpImageQueryLevels`, `OpImageQueryLod`, `OpImageQuerySamples`, `OpImageQuerySize`, `OpImageQuerySizeLod`, `OpImageSampleExplicitLod`, `OpImageSampleImplicitLod`, `OpInBoundsAccessChain`, `OpIsHelperInvocationEXT`, `OpIsInf`, `OpIsNan`, `OpKill`, `OpLine`, `OpLoad`, `OpLogicalAnd`, `OpLogicalEqual`, `OpLogicalNot`, `OpLogicalNotEqual`, `OpLogicalOr`, `OpLoopMerge`, `OpNoLine`, `OpNop`, `OpNot`, `OpPhi`, `OpQuantizeToF16`, `OpReturn`, `OpSDiv`, `OpSGreaterThan`, `OpSGreaterThanEqual`, `OpSLessThan`, `OpSLessThanEqual`, `OpSMod`, `OpSMulExtended`, `OpSNegate`, `OpSRem`, `OpSampledImage`, `OpSelect`, `OpSelectionMerge`, `OpShiftLeftLogical`, `OpShiftRightArithmetic`, `OpShiftRightLogical`, `OpStore`, `OpSwitch`, `OpUDiv`, `OpUGreaterThan`, `OpUGreaterThanEqual`, `OpULessThan`, `OpULessThanEqual`, `OpUMod`, `OpUMulExtended`, `OpUndef`, `OpUnreachable`, `OpVectorExtractDynamic`, `OpVectorInsertDynamic`, `OpVectorShuffle`, `OpVectorTimesScalar`.

The installed Khronos GLSL.std.450 instruction enumeration supplies the complete extended-instruction set below. Numerical semantics and undefined-case choices remain subject to the preceding notes.

| GLSL.std.450 status | Exact operations |
| --- | --- |
| Implemented (73) | `Acos`, `Acosh`, `Asin`, `Asinh`, `Atan`, `Atan2`, `Atanh`, `Ceil`, `Cos`, `Cosh`, `Cross`, `Degrees`, `Distance`, `Exp`, `Exp2`, `FAbs`, `FClamp`, `FMax`, `FMin`, `FMix`, `FSign`, `FaceForward`, `FindILsb`, `FindSMsb`, `FindUMsb`, `Floor`, `Fma`, `Fract`, `Frexp`, `FrexpStruct`, `InverseSqrt`, `Ldexp`, `Length`, `Log`, `Log2`, `Modf`, `ModfStruct`, `NClamp`, `NMax`, `NMin`, `Normalize`, `PackHalf2x16`, `PackSnorm2x16`, `PackSnorm4x8`, `PackUnorm2x16`, `PackUnorm4x8`, `Pow`, `Radians`, `Reflect`, `Refract`, `Round`, `RoundEven`, `SAbs`, `SClamp`, `SMax`, `SMin`, `SSign`, `Sin`, `Sinh`, `SmoothStep`, `Sqrt`, `Step`, `Tan`, `Tanh`, `Trunc`, `UClamp`, `UMax`, `UMin`, `UnpackHalf2x16`, `UnpackSnorm2x16`, `UnpackSnorm4x8`, `UnpackUnorm2x16`, `UnpackUnorm4x8` |
| Unsupported (8) | `Determinant`, `IMix`, `InterpolateAtCentroid`, `InterpolateAtOffset`, `InterpolateAtSample`, `MatrixInverse`, `PackDouble2x32`, `UnpackDouble2x32` |
