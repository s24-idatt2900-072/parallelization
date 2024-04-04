@group(0)
@binding(0)
// This is input buffer a
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
// This is input buffer b
var<storage, read> b: array<f32>;

@group(0)
@binding(2)
// This is the output buffer
var<storage, read_write> out: array<f32>;

const ilen: u32 = 841u;
const alen: u32 = 3800u;
const blen: u32 = 100000u;
const workgroup_size: vec3<u32> = vec3<u32>(16u, 16u, 1u);

@compute
@workgroup_size(16u, 16u, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    ) {
    // Thread id
    let tid = vec2(
            wid.x * workgroup_size.x + lid.x,
            wid.y * workgroup_size.y + lid.y
        );

    if tid.x >= alen || tid.y >= blen {
        return;
    }

    // Initial dot product
    var dot = 0.0;
    for (var i = 0u; i < ilen; i = i + 1u) {
        let a_index = tid.x * ilen + i;
        let b_index = tid.y * ilen + i;
        dot = dot + a[a_index] * b[b_index];
    }
    let out_index = (tid.x * blen) + tid.y;
    out[out_index] = dot;
}