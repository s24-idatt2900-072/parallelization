@group(0)
@binding(0)
var<storage, read> a: vec3<f32>;

@group(0)
@binding(1)
var<storage, read> b: vec3<f32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<f32>;

@compute
@workgroup_size(3, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    // Both dot and manul works
    //out[lid.x] = a[lid.x] * b[lid.x];
    out[0] = dot(a, b);
}
