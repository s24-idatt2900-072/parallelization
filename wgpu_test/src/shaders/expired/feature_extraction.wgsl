@group(0)
@binding(0)
var<storage, read> shapes: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> a: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> b: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> out: array<f32>;

var<workgroup> temp: array<f32, 36u>;

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
    // outer length a
    var olen_a = shapes[2];
    // processing chunk size
    var chunk = shapes[3];
    // filter chunk for maxpooling
    var filter_chunk = shapes[4];
    // thread id
    var tid = vec2(
            wid.x * workgroup_size.x + lid.x, 
            wid.y * workgroup_size.y + lid.y
        );
    if tid.x == 0u && tid.y == 0u{
        out[0] = out[0] + 1.0;
    }

    // Dot multiplication and 1 round sum
    ilen = dot_mult(tid, chunk, olen, ilen);
    // for linear paralelizm
    var max_x = num_workgroups.x * workgroup_size.x;

    for (var i = 0u; i < 0u; i = i + 1u) {
        workgroupBarrier();
        for (var j = 0u; j < chunk; j = j + 1u) {
            var index = (tid.x*chunk)+(tid.y*olen*olen_a*ilen*chunk)+j;
            if ilen > 1u{
                a[index] = 0.0;
            }
        }
        workgroupBarrier();
        
        ilen = sum(tid, chunk, max_x, ilen);

        workgroupBarrier();
        for (var j = 0u; j < chunk; j = j + 1u) {
            var index = (tid.x*chunk)+(tid.y*olen*olen_a*ilen*chunk)+j;
            out[index] = a[index];
        }
        workgroupBarrier();
    }


    ilen = filter_chunk;
    for (var i = 0u; i < 0u; i = i + 1u) {
        workgroupBarrier();
        var index = (tid.x)+(tid.y*olen*olen_a);
        a[index] = out[index];
        if ilen > 1u{
            out[index] = 0.0;
        }
        workgroupBarrier();

        ilen = max(tid, chunk, max_x, ilen);
    }
    temp[wid.x] = temp[wid.x] + a[lid.x] + b[lid.y];
    workgroupBarrier();
    out[wid.x] = temp[wid.x];
}

fn dot_mult(tid: vec2<u32>, chunk: u32, olen: u32, ilen: u32) -> u32 {
    var next_ilen = ilen;
    if ilen%chunk != 0 {
        next_ilen = next_ilen + 1u;
    }
    next_ilen = next_ilen/chunk;
    for (var i = 0u; i < chunk; i = i + 1u) {
        if tid.y < olen && tid.x*chunk+i < arrayLength(&a) {
            // a outer index
            var out_a = (tid.x*chunk+i)/ilen;
            // b start index + inner a index
            var b_index = tid.y*ilen + (tid.x*chunk+i)%ilen;
            // offset
            var offset = 0u;
            if ilen%chunk != 0 {
                offset = tid.y;//+tid.x;
            }
            // start output index for outer a
            var start = out_a*next_ilen*olen;
            // outer a output start index + b index offset
            var out_index = start + (b_index+offset)/chunk;
            // multiplication of a and similar local index b
            var dot = a[i+tid.x*chunk] * b[b_index];
            // sum to make dot product
            out[out_index] = out[out_index] + dot;
        }
    }
    return next_ilen;
}

fn sum(tid: vec2<u32>, chunk: u32, max_x: u32, il: u32) -> u32 {
    var ilen = il;
    if ilen <= 1u {
        return ilen;
    }
    for (var i = 0u; i < chunk; i = i + 1u) {
        // thread-x counting in chunks + thread-y * max-x
        var index = (tid.x*chunk) + (tid.y*max_x) + i;
        var offset = get_offset(index, chunk, ilen);
        var out_index = index/chunk + offset;
        a[out_index] = a[out_index] + out[index];
    }
    if ilen%chunk != 0 {
        ilen = ilen + 1u;
    }
    return ilen/chunk;
}

fn max(tid: vec2<u32>, chunk: u32, max_x: u32, il: u32) -> u32 {
    var ilen = il;
    if ilen <= 1u {
        return ilen;
    }
    for (var i = 0u; i < chunk; i = i + 1u) {
        // thread-x counting in chunks + thread-y * max-x
        var index = (tid.x*chunk) + (tid.y*max_x) + i;
        var offset = get_offset(index, chunk, ilen);
        var out_index = index/chunk + offset;
        if out[out_index] < a[index] {
            out[out_index] = a[index];
        }
    }
    if ilen%chunk != 0 {
        ilen = ilen + 1u;
    }
    return ilen/chunk;
}

fn get_offset(index: u32, chunk: u32, ilen: u32) -> u32{
    if ilen%chunk == 0 {
        return 0u;
    }
    var temp = index-ilen;
    if ilen > index {
        temp = 0u;
    }
    var crossed_count = temp/(ilen*chunk);
    if index >= ilen {
        crossed_count = crossed_count + 1u;
    }
    return crossed_count;
}
