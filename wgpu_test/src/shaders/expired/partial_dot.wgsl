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
@workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {  

    var image_index = lid.x;
    let length_image = shapes[1];
    let num_threads = 256u;


    var filter_index = wid.x * length_image + image_index;
    let filter_start_index = wid.x *length_image;

    let to = filter_start_index + length_image;
    
    var temp = 0.0f;
    
    while (filter_index < to) {
        temp += a[image_index] * b[filter_index];
        filter_index += num_threads;
        image_index += num_threads;
    }

    let thread_id = lid.x;
    let out_index = wid.x * num_threads + thread_id;
    let out_filter_start = wid.x * num_threads;
    out[out_index] = temp;
}    