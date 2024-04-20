@group(0)
@binding(0)
// This is input buffer features containing values that will be max pooled
var<storage, read> features: array<f32>;

@group(0)
@binding(1)
// This is the output buffer
var<storage, read_write> out: array<f32>;

const ilen: u32 = 500u;
const len: u32 = 100u;
const workgroup_size: vec3<u32> = vec3<u32>(256u, 1u, 1u);

@compute
@workgroup_size(256u, 1u, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    ) {
    // Thread id
    let tid = wid.x * workgroup_size.x + lid.x;

    if tid >= len {
        return;
    }

    var max = 0.0;
    for (var i = 0u; i < ilen; i = i + 1u) {
        let index = tid * ilen + i;
        if (features[index] > max) {
            max = features[index];
        }
    }
    out[tid] = max;
}