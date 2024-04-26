@group(0)
@binding(0)
var<storage, read> shapes: array<u32>;

@group(0)
@binding(1)
var<storage, read> a: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {  
    // thread id
    let idx = lid.x + wid.x * 256u;

    let start = idx * 256u;
    let end = start + 256u;

    var temp = 0.0f;
    // make foor loop from start to end
    for (var i = start; i < end; i++) {
        temp = temp + a[i];
    }

    out[idx] = temp;
}    
   

  