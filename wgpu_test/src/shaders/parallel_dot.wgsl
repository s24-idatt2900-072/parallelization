@group(0)
@binding(0)
// This is input buffer a
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
// This is input buffer b
var<storage, read> b: array<f32>;

@group(0)
@binding(2)
// This is the output buffer
var<storage, read_write> out: array<f32>;

// This is the temp buffer and unique to each workgroup
// That means each dot product has its own temp buffer
var<workgroup> temp: array<f32, 256>;

// This is the info: inner length, chunk size, next inner length
const info: vec3<u32> = vec3<u32>(841u, 10u, 85u);

@compute
@workgroup_size(253, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    ) {
    // Size of each workgroup
    let workgroup_size = vec3(253u, 1u, 1u);
    // inner length
    let ilen = info[0]; // 841
    // processing chunk size
    let chunk = info[1]; // 10
    // next inner length
    let next_ilen = info[2]; // 85
    // Length of a
    let alen = arrayLength(&a) / ilen;
    // Length of b
    let blen = arrayLength(&b) / ilen;
    // Work size for the workgroup
    let work_size = (chunk * workgroup_size.x) / ilen; // 3

    // Check if the workgroup is out of bound
    if wid.x * work_size >= alen 
        || wid.y >= blen 
    {
        return;
    }

    // Start and end index for the dot product
    let start = lid.x * chunk;
    let end = start + chunk; 
    var rest = 0u;
    if end >= ilen * work_size { 
        rest = chunk - (end % (ilen * work_size)); 
    }

    // Initial dot product
    for (var i = start; i < end - rest; i = i + 1u) {
        let a_index = wid.x * ilen * work_size + i;
        let b_index = wid.y * ilen + (i % ilen);
        let dot = a[a_index] * b[b_index];

        let temp_index= lid.x + i / ilen;
        temp[temp_index] = temp[temp_index] + dot;
    }
    workgroupBarrier();
    // Each local id will finish one dot product
    if lid.x < work_size {
        let start = lid.x * next_ilen;
        let end = start + next_ilen;
        // Last sum
        for (var i = start; i < end; i = i + 1u) {
            let out_index = (wid.x * blen * work_size) + (lid.x * blen) + wid.y;
            out[out_index] = out[out_index] + temp[i];
        }
    }
}
