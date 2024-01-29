@group(0)
@binding(0)
var<storage, read> a: vec3<f32>;

@group(0)
@binding(1)
var<storage, read_write> out: array<f32>;

@compute
@workgroup_size(6, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    for (var i = 0u; i < 3; i = i + 1u) {
        if a[i] > out[0] {
            out[0] = a[i];
        }
    }
}
