# Command-line translator

```sh
spirv-as --target-env vulkan1.1 tests/data/compute/ids.spvasm -o build/ids.spv
build/spirv2air_cli build/ids.spv build/ids.metallib --buffer 0:0:0
build/spirv2air_cli --version
```

The first two arguments are an input SPIR-V file and an output metallib file. The output entry point is always `main`, whatever the input entry point is called.

| Option | Meaning |
| --- | --- |
| `--buffer SET:BINDING:INDEX` | Compute buffer binding to Metal buffer index (repeat per resource) |
| `--texture SET:BINDING:INDEX` | Compute texture binding to Metal texture index (repeat per resource) |
| `--push-constant INDEX` | Metal buffer index for a compute push-constant block |
| `--function-constant-set SET` | Map that descriptor set's uint4 uniforms to Metal function constants ([binding contract](bindings.md#function-constants)) |
| `--storage-buffer SET:BINDING:OFFSET` | Graphics storage buffer whose address is at that argument-buffer offset (repeatable) |
| `--platform macos\|ios\|ios-simulator` | Target platform ([platforms](air.md#platforms)) |
| `--min-os MAJOR[.MINOR]` | Minimum OS version of the target |
| `--no-jump-thread` | Disable dispatch-loop jump threading |

Graphics shaders use the default layout in the [binding contract](bindings.md); the C++ API exposes all of it.

Exit codes: 0 success, 1 translation rejected, 2 argument or I/O error.

## Batch mode

`spirv2air_cli --manifest INPUT.txt OUTPUT_DIR` translates many modules at once.

- Each manifest row is `<source-key> <stage> <variant> <SPIR-V path>`, with whitespace-separated fields.
- Output files are named `<key>.<stage>.<variant>.metallib`, where `<key>` is the source key's basename up to its first dot. Output names must be unique.
- Batch mode uses the default graphics options. Only `--function-constant-set` and `--storage-buffer` may follow `--manifest INPUT.txt OUTPUT_DIR`.
- It reports translation totals and diagnostics, and returns 1 if any module fails.
- The older form `spirv2air_cli INPUT.txt OUTPUT_DIR` is still accepted for `.txt` manifests.
