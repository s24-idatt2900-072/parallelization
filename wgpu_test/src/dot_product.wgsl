@group(0)
@binding(0)
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
var<storage, read> b: array<f32>;

@group(0)
@binding(2)
var<storage, read> shapes: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> out: array<f32>;

@compute
@workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    // inner length
    var ilen = shapes[0];
    // outer length
    var olen = shapes[1];
    // id x
    var idx = wid.x * 16 + lid.x;
    // id y
    var idy = wid.y * 16 + lid.y;
    // from index
    var fr = idx * ilen;
    // to index
    var to = fr + ilen;
    // b index
    var ib = idy * ilen;
    if fr > arrayLength(&a) || to > arrayLength(&a) {
        return;
    } else if ib > arrayLength(&b) || ib + ilen > arrayLength(&b) {
        return;
    }
    
    var dot = 0.0;
    for (var i = fr; i < to; i = i + 1u) {
        dot = dot + a[i] * b[ib];
        ib = ib + 1u;
    }
    out[idx * olen + idy] = dot;
    //workgroupBarrier();
}
