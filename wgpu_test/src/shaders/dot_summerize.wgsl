@group(0)
@binding(0)
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
var<storage, read> real: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> b: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> staging_filter: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> result: array<f32>;



@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {  

    var image_index = lid.x;
    let length_image = 841u;
    let num_threads = 256u;


    var filter_index = wid.x * length_image + image_index;
    let filter_start_index = wid.x * length_image;
    let to = filter_start_index + length_image;
    
    // Make D and write it to b
    while (filter_index < to) {
        b[filter_index] = a[image_index] * b[filter_index];
        filter_index += num_threads;
        image_index += num_threads;
    }
    
    // sync threads
    workgroupBarrier();

    image_index = lid.x;
    filter_index = wid.x * length_image + image_index;

    var temp = 0.0f;
    // D * Re and write it to staging_filter which is summed to 256 values
    while (filter_index < to) {
        temp += real[filter_index] * b[filter_index];
        filter_index += num_threads;
        image_index += num_threads;
    }

    image_index = lid.x;
    let staging_filter_index = wid.x * num_threads + image_index;
    var staging_filter_filter_start = wid.x * num_threads;
    staging_filter[staging_filter_index] = temp;    

    // sync threads
    workgroupBarrier();

    // get size of workgroup 
    var size = num_threads/2u;
    
    // sums up D * Re in staging_filter
    while (size != 0) {
        if (staging_filter_index < staging_filter_filter_start + size) {
            staging_filter[staging_filter_index] += staging_filter[staging_filter_index + size];    
        }

        workgroupBarrier();
        size = size / 2;
    }

    filter_index = wid.x * length_image + image_index;
    temp = 0.0f;

    // D * D and write it to B
    while (filter_index < to) {
        temp += b[filter_index] * b[filter_index];
        filter_index += num_threads;
    }

    image_index = lid.x;
    let staging_filter_filter = wid.x * length_image + image_index;
    staging_filter_filter_start = wid.x * length_image;
    b[staging_filter_filter] = temp;

    // sync threads
    workgroupBarrier();

    // sums up D * D in b
    size = num_threads/2u;
    while (size != 0) {
        if (staging_filter_filter < staging_filter_filter_start + size) {
            b[staging_filter_filter] += b[staging_filter_filter + size];    
        }

        workgroupBarrier();
        size = size / 2;
    }
    
    // makes the cosine similarity in staging_filter with spaces of 256
    if (lid.x == 0) {
        result[wid.x] = staging_filter[staging_filter_index]/sqrt(b[staging_filter_filter]);
    }
}    