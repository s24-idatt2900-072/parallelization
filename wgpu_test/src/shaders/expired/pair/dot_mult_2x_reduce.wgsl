@group(0)
@binding(0)
// This is the shape buffer and other parameters
var<storage, read> shapes: array<u32>;

@group(0)
@binding(1)
// This is the current dispatch number
var<storage, read> dispatch: u32;

@group(0)
@binding(2)
// This is the a array
var<storage, read> a: array<f32>;

@group(0)
@binding(3)
// This is the b array
var<storage, read> b: array<f32>;

@group(0)
@binding(4)
// This is the output array
var<storage, read_write> out: array<f32>;

// processing chunk size
const chunk = 2u;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    ) {
    // Size of each workgroup
    var workgroup_size = vec3(16u, 16u, 1u);
    // inner length
    var ilen = shapes[0];
    // outer length b
    var olen = shapes[1];
    // thread id
    var tid = vec2(
            wid.x * workgroup_size.x + lid.x,
            wid.y * workgroup_size.y + lid.y
        );
    // Dot multiplication and 1 round sum
    dot_mult(tid, olen, ilen);
}

fn dot_mult(tid: vec2<u32>, olen: u32, ilen: u32) {
    var next_ilen = shapes[4];
    var offset = 0u;
    if ilen%chunk != 0 {
        offset = tid.y;
    }
    if tid.y < olen {
        // a index
        var a_index = tid.x*chunk;
        dot(tid, a_index, olen, ilen, offset, next_ilen);
        dot(tid, a_index+1u, olen, ilen, offset, next_ilen);
    }
}

fn dot(tid: vec2<u32>, a_index: u32, olen: u32, ilen: u32, off: u32, next_ilen: u32) {
    // a outer index
    var out_a = a_index/ilen;
    var offset = off;
    if out_a % chunk != 0u {
        // + Odd offset
        offset = offset + 1u;
    }
    // b start index + inner a index
    var b_index = tid.y*ilen + a_index%ilen;
    // start output index for outer a
    var start = out_a*next_ilen*olen;
    // outer a output start index + b index offset
    var out_index = start + (b_index+offset)/chunk;
    // a work size
    var a_work_size = shapes[7];
    // offset for a
    var a_offset = a_work_size * dispatch * ilen;
    // multiplication of a and similar local index b
    var dot = a[a_index + a_offset] * b[b_index];
    // sum to make dot product
    out[out_index] = out[out_index] + dot;
}
