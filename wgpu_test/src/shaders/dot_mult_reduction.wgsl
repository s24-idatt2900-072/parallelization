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
@workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
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
    // thread id
    var tid = vec2(
            wid.x * workgroup_size.x + lid.x, 
            wid.y * workgroup_size.y + lid.y
        );
    // Dot multiplication and 1 round sum
    dot_mult(tid, chunk, olen, ilen);
}

fn dot_mult(tid: vec2<u32>, chunk: u32, olen: u32, ilen: u32) {
    var next_ilen = ilen;
    // Partalls offset ?
    var par_off = 0u;
    while next_ilen%chunk != 0 {
        next_ilen = next_ilen + 1u;
        par_off = par_off + 1u;
    }
    next_ilen = next_ilen/chunk;
    var offset = 0u;
    if ilen%chunk != 0 {
        offset = tid.y;
    }
    if tid.y < olen {
        for (var i = 0u; i < chunk; i = i + 1u) {
            // a index
            var a_index = tid.x*chunk+i;
            // a outer index
            var out_a = a_index/ilen;
            if out_a%2 != 0u && a_index%chunk == 0u{
                offset = offset /*+ par_off;*/+ 1u; // usikker på hva som er riktig her
                // ved chunk = 2, vil det være plus 1u uansett :)
            }
            // b start index + inner a index
            var b_index = tid.y*ilen + a_index%ilen;
            // start output index for outer a
            var start = out_a*next_ilen*olen;
            // outer a output start index + b index offset
            var out_index = start + (b_index+offset)/chunk;
            // multiplication of a and similar local index b
            var dot = a[a_index] * b[b_index];
            // sum to make dot product
            out[out_index] = out[out_index] + dot;
        }
    }
}
