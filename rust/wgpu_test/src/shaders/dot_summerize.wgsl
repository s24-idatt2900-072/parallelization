@group(0)
@binding(0)
var<storage, read> a: array<f32>;

@group(0)
@binding(1)
var<storage, read> real: array<f32>;

@group(0)
@binding(2)
var<storage, read> offset: array<u32>;

@group(0)
@binding(3)
var<storage, read_write> b: array<f32>;

@group(0)
@binding(4)
var<storage, read_write> staging_filter: array<f32>;

@group(0)
@binding(5)
var<storage, read_write> result: array<f32>;



@compute
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {  
    let length_image = 841u;
    var image_index =  offset[0] * length_image + lid.x ;
    let num_threads = 256u;


    var filter_index = wid.x * length_image + lid.x;
    let filter_start_index = wid.x * length_image;
    let to = filter_start_index + length_image;
    
    // Make D and write it to b
    while (filter_index < to) {
        let abs = b[filter_index];
        let img = a[image_index];
        b[filter_index] = abs * img ; //* b[filter_index];
        filter_index += num_threads;
        image_index += num_threads;
    }
    
    // sync threads
    workgroupBarrier();

    filter_index = wid.x * length_image + lid.x;

    var temp = 0.0f;
    // D * Re and write it to staging_filter which is summed to 256 values
    while (filter_index < to) {
        temp += real[filter_index] * b[filter_index];
        filter_index += num_threads;
    }

    let staging_filter_index = wid.x * num_threads + lid.x;
    var staging_filter_start = wid.x * num_threads;
    staging_filter[staging_filter_index] = temp;    

    // sync threads
    workgroupBarrier();

    // get size of workgroup 
    var size = num_threads/2u;
    
    // sums up D * Re in staging_filter
    while (size != 0) {
        if (staging_filter_index < staging_filter_start + size) {
            staging_filter[staging_filter_index] += staging_filter[staging_filter_index + size];    
        }

        workgroupBarrier();
        size = size / 2;
    }

    filter_index = wid.x * length_image + lid.x;
    temp = 0.0f;

    // D * D and write it to B
    while (filter_index < to) {
        temp += b[filter_index] * b[filter_index];
        filter_index += num_threads;
    }

    filter_index = wid.x * length_image + lid.x;
    staging_filter_start = wid.x * length_image;
    b[filter_index] = temp;

    // sync threads
    workgroupBarrier();

    // sums up D * D in b
    size = num_threads/2u;
    while (size != 0) {
        if (filter_index < staging_filter_start + size) {
            b[filter_index] += b[filter_index + size];    
        }

        workgroupBarrier();
        size = size / 2;
    }
    
    // makes the cosine similarity in staging_filter with spaces of 256
    if (lid.x == 0) {
        let off = offset[0] * arrayLength(&real) / length_image;
        result[wid.x + off] = staging_filter[staging_filter_index]/sqrt(b[filter_index]);
    }
}    