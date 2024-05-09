@group(0)
@binding(0)
// This is input buffer image, the image to be convolved
var<storage, read> image: array<f32>;

@group(0)
@binding(1)
// This is input buffer re, the real part of the filter
var<storage, read> re: array<f32>;

@group(0)
@binding(2)
// This is input buffer abs, the absolute part of the filter
var<storage, read> abs: array<f32>;

@group(0)
@binding(3)
// This is the output buffer
var<storage, read_write> out: array<f32>;

// This is the temp buffer and unique to each workgroup
// It is used to store the normalized image and filter
var<workgroup> norm_d: array<f32, 255>;

// This is the temp buffer and unique to each workgroup
// It is used to store the dot product of the image and filter
var<workgroup> temp_dot: array<f32, 255>;

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
    let workgroup_size = vec3(253, 1u, 1u);
    // inner length
    let ilen = info[0]; // 841
    // processing chunk size
    let chunk = info[1]; // 10
    // next inner length
    let next_ilen = info[2]; // 85
    // Length of a
    let alen = arrayLength(&image) / ilen;
    // Length of b
    let blen = arrayLength(&re) / ilen;
    // Work size for the workgroup
    let work_size = (chunk * workgroup_size.x) / ilen; // 3
    // Thread id
    let tid = vec2(
            wid.x,
            wid.y * work_size
        );
    // Check if the workgroup is out of bound
    if tid.x >= alen 
        || tid.y >= blen 
    {
        return;
    }

    // Start and end index for the dot product
    let start = lid.x * chunk;
    var end = start + chunk; 
    let over = end % (ilen * work_size);
    end = end - over * (end / (ilen * work_size));

    // Initial dot product
    for (var i = start; i < end; i = i + 1u) {
        let image_index = tid.x * ilen + (i % ilen);
        let filter_index = tid.y * ilen + i;

        let d = image[image_index] * abs[filter_index];
        let dot = d * re[filter_index];
        let norm_dot = d * d;

        let temp_index= lid.x + i / ilen;
        temp_dot[temp_index] = temp_dot[temp_index] + dot;
        norm_d[temp_index] = norm_d[temp_index] + norm_dot;
    }

    workgroupBarrier();
    
    // Each local id will finish one dot product
    if lid.x < work_size && lid.x + tid.y < blen {
        let start = lid.x * next_ilen;
        let end = start + next_ilen;
        // Last sum
        var dot = 0.0;
        var norm_dot = 0.0;
        for (var i = start; i < end; i = i + 1u) {
            dot = dot + temp_dot[i];
            norm_dot = norm_dot + norm_d[i];
        }
        let out_index = (tid.x * blen) + lid.x + tid.y;
        // The dot product divided by the nomalized image and filter
        out[out_index] = dot / sqrt(norm_dot);
    }
}
