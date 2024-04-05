@group(0)
@binding(0)
var<storage, read> shapes: array<u32>;

@group(0)
@binding(1)
var<storage, read> a: array<f32>;

@group(0)
@binding(2)
var<storage, read> b: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> out: array<f32>;

@compute
@workgroup_size(4, 64, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    ) {
    // Size of each workgroup
    var workgroup_size = vec3<u32>(4, 64, 1);
    // inner length
    var ilen = shapes[0];
    // outer length
    var olen = shapes[1];
    // thread id
    var tid = vec2(
            wid.x * workgroup_size.x + lid.x, 
            wid.y * workgroup_size.y + lid.y
        );

    if tid.y < olen {
        // a index
        var a_index = tid.x;
        // a outer index
        var out_a = a_index/ilen;
        // b start index + inner a index
        var b_index = tid.y*ilen + a_index%ilen;
        // start output index for outer a
        var start = out_a*ilen*olen;
        // outer a output start index + b index offset
        var out_index = start + b_index;
        // multiplication of a and similar local index b
        var dot = a[a_index] * b[b_index];
        // sum to make dot product
        out[out_index] = out[out_index] + dot;
    }
}    