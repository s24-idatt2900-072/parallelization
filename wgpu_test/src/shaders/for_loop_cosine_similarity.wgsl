@group(0)
@binding(0)
// This is input buffer image, the image to be convolved
var<storage, read> image: array<f32>;

@group(0)
@binding(1)
// This is input buffer re, the real part of the filter
var<storage, read> re: array<f32>;

@group(0)
@binding(2)
// This is input buffer abs, the absolute part of the filter
var<storage, read> abs: array<f32>;

@group(0)
@binding(3)
// This is the output buffer
var<storage, read_write> out: array<f32>;

const ilen: u32 = 841u;
const alen: u32 = 10u;
const blen: u32 = 10u;
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

    var dot = 0.0;
    var norm = 0.0;
    for (var i = 0u; i < ilen; i = i + 1u) {
        let image_index = tid.x * ilen + i;
        let filter_index = tid.y * ilen + i;
        let d = image[image_index] * abs[filter_index];
        dot = dot + d * re[filter_index];
        norm = norm + d * d;
    }
    let out_index = (tid.x * blen) + tid.y;
    out[out_index] = dot / sqrt(norm);
}