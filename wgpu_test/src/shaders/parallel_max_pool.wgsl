@group(0)
@binding(0)
// This is input buffer features containing values that will be max pooled
var<storage, read> features: array<f32>;

@group(0)
@binding(1)
// This is the output buffer
var<storage, read_write> out: array<f32>;

// This is the temp buffer and unique to each workgroup
// That means each dot product has its own temp buffer
var<workgroup> temp: array<f32, 250>;

// This is the info: inner length, chunk size, next inner length
const info: vec3<u32> = vec3<u32>(500u, 10u, 50u);

@compute
@workgroup_size(250, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
    ) {
    // Size of each workgroup
    let workgroup_size = vec3(250, 1u, 1u);
    // inner length
    let ilen = info[0]; 
    // processing chunk size
    let chunk = info[1];
    // next inner length
    let next_ilen = info[2]; 
    // Length of a
    let len = arrayLength(&features) / ilen;
    // Work size for the workgroup
    let work_size = (chunk * workgroup_size.x) / ilen; // 5
    // Thread id
    let tid = wid.x * work_size;
    // Check if the workgroup is out of bound
    if tid >= len {
        return;
    }

    // Start and end index for the max pooling
    let start = lid.x * chunk;
    var end = start + chunk; 
    let over = end % (ilen * work_size);
    end = end - over * (end / (ilen * work_size));

    // Initial max pooling
    for (var i = start; i < end; i = i + 1u) {
        let index = tid * ilen + i;
        let temp_index= lid.x + i / ilen;
        if temp[temp_index] < features[index] {
            temp[temp_index] = features[index];
        }
    }

    workgroupBarrier();
    
    // Each local id will finish one max pooling
    if lid.x < work_size {
        let start = lid.x * next_ilen;
        let end = start + next_ilen;
        // Last max pooling
        let out_index = tid + lid.x;
        for (var i = start; i < end; i = i + 1u) {
            if out[out_index] < temp[i] {
                out[out_index] = temp[i];
            }
        }
    }
}
