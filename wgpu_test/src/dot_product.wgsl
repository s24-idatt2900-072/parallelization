@group(0)
@binding(0)
var<storage, read> a: vec3<i32>;

@group(0)
@binding(1)
var<storage, read> b: vec3<i32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<i32>;

@compute
@workgroup_size(6, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    // bouth dot and manul works
    // issue with float parsing atm
    out[lid.x] = /*dot(a, b);*/a[lid.x] * b[lid.x];
}
