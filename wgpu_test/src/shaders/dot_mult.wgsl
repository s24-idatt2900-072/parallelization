@group(0)
@binding(0)
var<storage, read> shapes: array<u32>;

@group(0)
@binding(1)
var<storage, read_write> a: array<u32>;

@group(0)
@binding(2)
var<storage, read_write> b: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> out: array<u32>;

@compute
@workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    ) {
    // Size of each workgroup
    var workgroup_size = vec3<u32>(16, 16, 1);
    // inner length
    var ilen = shapes[0];
    // outer length
    var olen = shapes[1];
    // index counter for a
    var idx = wid.x * workgroup_size.x + lid.x;
    // id for outer b
    var idy = wid.y * workgroup_size.y + lid.y;

    // a outer index
    var out_a = idx/ilen; // bilde id [0-n] bilder
    // out start index
    var start = out_a*olen*ilen; // bilde id * multiplikasjons lengden for dot produkt gange antall filter
    // local a inner index
    var in_a = idx%ilen; // Den indre bilde indexen. f.eks: bilde 1 index 3: [[0,1,2,3],[0,1,2,X]]
    // b start index
    var b_start = idy*ilen; // idy representerer filteret og ganget med indre lengde gir start for filter x
    // b index
    var b_index = in_a + b_start; // indre filter indexen
    // out index for multiplication
    var out_index = start + b_index; // bildet id sin start pluss filterindexen
    
    if idy < olen && idx < arrayLength(&a) {
        // multiplication of a and similar local index b
        var dot = a[idx] * b[b_index];
        out[out_index] = dot;
    }
}    