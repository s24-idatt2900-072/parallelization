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
// This is the input buffer
var<storage, read> input: array<f32>;

@group(0)
@binding(3)
// This is the output buffer
var<storage, read_write> out: array<f32>;

// This is the temp buffer
var<workgroup> temp: array<f32, 256>;

@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    ) {
    // Size of each workgroup
    var workgroup_size = vec3(256u, 1u, 1u);
    // inner length
    var ilen = shapes[4]; // 421
    // processing chunk size
    var chunk = shapes[3]; // 5

    var work_size = (chunk * workgroup_size.x) / ilen; // 3
    // if list is to large for 1 dispatch, the later dispatches gets a offset
    var offset = num_workgroups.x * work_size * dispatch * ilen; // 65_536 * 3 * [0-5] = 0
    var next_ilen = shapes[5]; // 85

    for (var i = 0u; i < chunk; i = i + 1u) {
        var local_index = lid.x * chunk + i; // 0 - 1280 || 1260, lid = 252
        if local_index >= ilen*work_size {  // local >= 1263 0 indexert med 1263 element resulterer i 1262 som er siste index
            break;
        }
        var temp_index= lid.x + local_index / ilen;
        var input_index = wid.x * ilen * work_size + local_index + offset;
        temp[temp_index] = temp[temp_index] + input[input_index];
    }
    workgroupBarrier();
    // Last sum
    if lid.x < work_size {
        for (var i = lid.x*next_ilen; i < (lid.x+1u)*next_ilen; i = i + 1u) {
            var offset = num_workgroups.x * work_size * dispatch;
            var out_index = wid.x*work_size + lid.x + offset;
            out[out_index] = out[out_index] + temp[i];
        }
    }
}
