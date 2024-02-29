@group(0)
@binding(0)
var<storage, read> shapes: array<u32>;

@group(0)
@binding(1)
var<storage, read> input: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> out: array<f32>;

var<workgroup> temp: array<f32, 256>;

@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    ) {
    // Size of each workgroup
    var workgroup_size = vec3(256u, 1u, 1u);
    // inner length
    var ilen = shapes[0];
    ilen=421u;
    // processing chunk size
    var chunk = shapes[3]; // 5

    var work_size = (chunk * workgroup_size.x) / ilen;
    var next_ilen = get_next_len(chunk, ilen);

    for (var i = 0u; i < chunk; i = i + 1u) {
        var local_index = lid.x * chunk + i;
        if local_index >= ilen*work_size { 
            break;
        }
        var temp_index= lid.x + local_index / ilen;
        var input_index = wid.x * ilen * work_size + lid.x * chunk + i;
        temp[temp_index] = temp[temp_index] + input[input_index];
    }
    workgroupBarrier();
    // Last sum
    if lid.x < work_size {
        for (var i = lid.x*next_ilen; i < (lid.x+1u)*next_ilen; i = i + 1u) {
            var out_index = wid.x*work_size + lid.x;
            out[out_index] = out[out_index] + temp[i];
        }
    }
}

fn get_next_len(chunk: u32, ilen: u32) -> u32 {
    var next_ilen = ilen;
    while next_ilen%chunk != 0 {
        next_ilen = next_ilen + 1u;
    }
    return next_ilen/chunk;
}
