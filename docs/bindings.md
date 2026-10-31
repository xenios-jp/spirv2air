# Graphics binding contract

This is the host interface implemented by spirv2air, not a complete Vulkan descriptor model. `Options::graphics` is a `GraphicsBindings` value. All defaults preserve the original integration. Compute instead uses `Options::bindings`: the SPIR-V resource kind selects `ResourceBinding::buffer` or `texture`; a missing pair is an error. Compute push constants use the separate `pushConstantBuffer` (default -1, meaning absent).

## Graphics inputs

| Resource | Default SPIR-V set/binding | Metal location or pointer source |
|---|---|---|
| Uniform buffers | set 1, bindings 0–4 | Top-level buffer pointer at byte `8 * binding` |
| Shared read-only storage | set 0, binding 0, NonWritable member | Device pointer at top-level byte 48 |
| Shared writable storage | set 0, binding 0 | Device pointer at top-level byte 56 |
| Fragment occlusion counter | set 0, binding 1 | Direct device buffer 7 |
| Fragment raster-ordered storage | set 0, binding 2 | Direct device buffer `rasterOrderedBuffer`, raster order group 0 (disabled by default) |
| Vertex/tessellation textures and samplers | set 2 | Per-binding indices through top-level byte 64 |
| Fragment textures and samplers | set 3 | Per-binding indices through top-level byte 72 |
| Tessellation control points | One float array input by Location | Direct device buffer 6 |

The top-level buffer is bound at `argumentBuffer=2`. The resource and sampler heaps use `resourceHeapBuffer=0` and `samplerHeapBuffer=1`. Pointer fields are 64-bit little-endian values, 8-byte aligned. The host allocates enough top-level storage for every configured slot it uses; the compiler does not allocate or populate these buffers.

Uniform slots use `constantBaseOffset=0`, `constantStride=8`, `constantBufferCount=5`, and `constantsSet=1`. Shared-memory selection uses `sharedMemorySet=0`, `sharedMemoryBinding=0`, `sharedMemoryReadOffset=48`, and `sharedMemoryWriteOffset=56`. The counter uses `occlusionCounterSet=0`, `occlusionCounterBinding=1`, and `occlusionCounterBuffer=7`; setting the buffer negative disables its special mapping. A module requesting a disabled input is rejected.

`rasterOrderedBuffer` (default -1, disabled; xenia's EDRAM uses 8) maps the StorageBuffer at `rasterOrderedSet=0`, `rasterOrderedBinding=2` of a fragment shader to that direct device buffer, declared in raster order group 0 like an MSL `[[raster_order_group(0)]]` buffer argument, so the fragments covering a pixel access it in primitive order. This is how `FragmentShaderPixelInterlockEXT` with `PixelInterlockOrderedEXT` translates: `OpBeginInvocationInterlockEXT` and `OpEndInvocationInterlockEXT` emit nothing, the raster order group orders the accesses instead, so the interlocked accesses must be to this buffer. A module declaring pixel interlock while the buffer is disabled is rejected. The host needs a device whose `areRasterOrderGroupsSupported` is true. tests/ordered_gpu.mm folds 64 overlapping primitives into every pixel non-atomically and checks primitive order against MSL.

Stage resource sets are `vertexTexturesSet=2` and `fragmentTexturesSet=3`. Their index-buffer pointers are read from `vertexTextureIndicesOffset=64` or `fragmentTextureIndicesOffset=72`. Each binding's index record starts at `binding * textureIndexStride` (default 8); its texture and sampler indices are uint32 values at `textureIndexOffset=0` and `samplerIndexOffset=4`.

Each index selects a heap record with `descriptorStride=24`. The texture handle is an 8-byte field at `textureHandleOffset=8`; the sampler handle is an 8-byte field at `samplerHandleOffset=0` in the sampler heap. The unused fields in the default resource record can hold a buffer address and metadata; translated shaders never read them. In particular no per-texture minimum LOD or LOD bias is taken from the record: a SPIR-V `Bias` or `Lod` image operand is passed to the sample intrinsic, the sampler state's own `lodMinClamp` and `lodMaxClamp` (and `lodBias`, macOS 26 / iOS 26 and later) apply, and the `MinLod` operand is rejected. A host that needs a per-texture minimum LOD or bias puts it into the sampler state, or into the shader as a `Bias` operand. tests/tex_gpu.mm checks `Bias` and `Lod` on a mipmapped texture against MSL `bias()` and `level()`. The host must supply valid GPU handles, keep resources alive and resident, and avoid index/offset overflow or overlapping bindings. Changing an offset does not change the 64-bit pointer or 32-bit index representation.

`controlPointBuffer=6` contains consecutive floats: control point k of patch p is element `p * N + k`. N is the input array length. A negative buffer disables this input. Metal tessellation factors, spacing, winding, and draw setup belong to the host pipeline.

## Stage interface and legacy semantics

Graphics varyings use `user(texcoordN)` for SPIR-V Location N. Vertex input is fetched by shader buffer loads; general vertex attributes are not mapped. Varyings are float, int or uint scalars and vectors (`air.arg_type_name` follows the SPIR-V signedness); Flat is `air.flat` on fragment inputs, and integer fragment inputs must be Flat. Fragment color outputs are float, int or uint scalars and vectors (the host's pixel format needs at least as many components). The emitted function name is `main`.

`vertexIndexRelativeToBase` defaults true: VertexIndex is `vertex_id - base_vertex`. False uses native `vertex_id` without declaring a base-vertex argument. `positionInfToNan` defaults true: position xyz with all exponent bits set becomes a quiet NaN; w is untouched. False returns the unmodified position. `jumpThread` defaults true and rewrites constant dispatch edges after promotion to SSA. It is independent of resource layout and applies to compute as well as graphics.

## Texture view types

The handle loaded from a heap record has the Metal type of the SPIR-V image type, so the host binds a view of that `MTLTextureType`: `Dim2D` with `Arrayed=1` is `MTLTextureType2DArray` and with `Arrayed=0` is `MTLTextureType2D`; `DimCube` is `MTLTextureTypeCube` or, arrayed, `MTLTextureTypeCubeArray`; `Dim3D` (not arrayed) is `MTLTextureType3D`. The array layer is `clamp(RoundEven(coordinate), 0, layers - 1)`; a NaN layer selects layer 0. xenia declares 2D fetches arrayed, cube fetches non-arrayed and 3D fetches as 3D, and binds the matching views.

## Non-finite positions

Measured with an MSL vertex function on one Apple GPU (macOS 27, one full-target triangle drawn with one poisoned vertex, or all three): a NaN in any position component culls the primitive, and so does +Inf in x, y or z. A -Inf in x or y at one vertex leaves the triangle drawn (the clipper treats it as a far-out coordinate); with all three vertices poisoned nothing is drawn. -Inf in z culls. For w, NaN culls, +Inf draws part of the target (22 of 64 pixels) and -Inf culls. `positionInfToNan` therefore turns an Inf in x, y or z into NaN so that every non-finite coordinate culls the same way, and leaves w alone. Whether a host wants that is a policy of the original integration, not a Metal requirement.

## Function constants

`functionConstantSet` (default -1, disabled; xenia's fetch state uses 9) maps the Uniform variables of one descriptor set to Metal function constants instead of argument-buffer slots. Each variable must be a block with one `uint4` member. Binding N becomes the `uint4` function constant `spirv2air_fc_<N>` with function constant index N; the host sets it with `MTLFunctionConstantValues` (`setConstantValue:type:withName:` with `MTLDataTypeUInt4`, or `atIndex:`) and creates the function with `newFunctionWithName:constantValues:error:`. Shaders may load the block, member 0, or one component of it; other uses are rejected. Constants the translated shader does not read are not declared, and values set for undeclared constants are ignored. The CLI flag is `--function-constant-set SET`, also accepted after `--manifest INPUT OUTPUT_DIR`. The encoding is described in [air.md](air.md#function-constants).

## Extra storage buffers

`graphics.storageBuffers` lists further StorageBuffer variables a graphics shader may use, each `{set, binding, argumentBufferOffset}`: the buffer's device address is the 8-byte pointer at that byte offset of the top-level argument buffer (like the shared-memory slots at 48 and 56). Loads and 32-bit integer atomics (add, and, or, signed and unsigned min and max) work in vertex and fragment shaders; Apple's compiler writes the same `air.atomic.global.*` calls for an MSL vertex function. Xenia's tile cull bounds (an `int` array in set 0, binding 4, argument buffer root parameter 14) use `{0, 4, 112}`. A StorageBuffer variable that matches no entry, shared memory or the counter is rejected. The CLI flag is `--storage-buffer SET:BINDING:OFFSET` (repeatable, also accepted after `--manifest`).

