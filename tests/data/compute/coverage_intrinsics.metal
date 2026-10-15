// Own MSL signature probes for the coverage lowerings.
// xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fno-fast-math -S -emit-llvm
//       coverage_intrinsics.metal -o coverage_intrinsics.ll
#include <metal_stdlib>
using namespace metal;
kernel void math_probe(device float* o [[buffer(0)]], device uint* u [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
  float x = o[i], y = o[i + 1], z = o[i + 2];
  o[i] = exp(x) + log(x) + pow(x, y) + fma(x, y, z) + fmod(x, y);
  u[i] = reverse_bits(u[i]) + popcount(u[i]);
  o[i + 1] = unpack_snorm4x8_to_float(u[i]).x + unpack_unorm4x8_to_float(u[i]).y;
  o[i + 2] = unpack_snorm2x16_to_float(u[i]).x + unpack_unorm2x16_to_float(u[i]).y;
  u[i + 1] = pack_float_to_snorm4x8(float4(x)) + pack_float_to_unorm4x8(float4(x));
  u[i + 2] = pack_float_to_snorm2x16(float2(x)) + pack_float_to_unorm2x16(float2(x));
  float integral;
  int exponent;
  o[i + 3] = modf(x, integral) + integral;
  o[i + 4] = frexp(x, exponent) + ldexp(y, exponent);
}
kernel void sync_probe(device atomic_uint* u [[buffer(0)]], threadgroup atomic_uint* s [[threadgroup(0)]]) {
  uint e = 7;
  bool r = atomic_compare_exchange_weak_explicit(u, &e, 9, memory_order_relaxed, memory_order_relaxed);
  atomic_store_explicit(u, atomic_load_explicit(s, memory_order_relaxed) + uint(r), memory_order_relaxed);
}
kernel void query_probe(texture2d<float> t [[texture(0)]], texture2d_ms<float> ms [[texture(1)]],
                        device uint* o [[buffer(0)]]) {
  o[0] = t.get_width(1);
  o[1] = t.get_height(1);
  o[2] = t.get_num_mip_levels();
  o[3] = ms.get_num_samples();
  o[4] = ms.get_width();
  o[5] = ms.get_height();
}
kernel void more_sync(device atomic_uint* u [[buffer(0)]], threadgroup atomic_uint* s [[threadgroup(0)]]) {
  uint e = 7;
  while (!atomic_compare_exchange_weak_explicit(s, &e, 9, memory_order_relaxed, memory_order_relaxed)) {
  }
}
kernel void more_query(texture2d_array<float> a [[texture(0)]], texturecube_array<float> c [[texture(1)]],
                       texture3d<float> t [[texture(2)]], device uint* o [[buffer(0)]]) {
  o[0] = a.get_width(1) + a.get_height(1) + a.get_array_size() + a.get_num_mip_levels();
  o[1] = c.get_width(1) + c.get_height(1) + c.get_array_size() + c.get_num_mip_levels();
  o[2] = t.get_width(1) + t.get_height(1) + t.get_depth(1) + t.get_num_mip_levels();
}
kernel void mulhi_probe(device uint* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  uint a = o[i], b = o[i + 1];
  o[i + 2] = mulhi(a, b);
  o[i + 3] = uint(mulhi(int(a), int(b)));
}
