@group(0)
@binding(0)
var<storage, read> images: array<f32>;

@group(0)
@binding(1)
var<storage, read> real: array<f32>;

@group(0)
@binding(2)
var<storage, read> abs: array<f32>;

@group(0)
@binding(3)
var<storage, read_write> out: array<f32>;

var<workgroup> staging_filter: array<f32, 64>;

var <workgroup> d_staging: array<f32, 64>;



@compute
@workgroup_size(64, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {  
    let length_image = 841u;
    let offset_image_id = wid.y;
    var image_index = lid.x + offset_image_id * length_image; // tar lengden til bilde som offset,
    // og ganger med hvilket bilde det er for å få riktig index på tråden
    let num_threads = 64u;
    let num_filters = arrayLength(&real) / length_image;


    var filter_index = wid.x * length_image + lid.x; // hvilken filter den skal starte på
    //wid.x represnterer hvilket filter den skal starte på
    let filter_start_index = wid.x * length_image; // starten på filteret
    var to = filter_start_index + length_image; // slutten på filteret

    let off_set_d_buffer = offset_image_id * (num_filters * length_image); // offset for d_staging
    // siden det er med alle bilder og alle filtre må en offsette med antall bilder og antall filtre i 
    // staging bufferet
    
    // Make D and write it to abs
    var temp_d = 0.0f;
    var temp_re_d = 0.0f;
    while (filter_index < to) {
        // d_staging[off_set_d_buffer + filter_index] = images[image_index] * abs[filter_index];
        let temp = images[image_index] * abs[filter_index];
        temp_re_d += real[filter_index] * temp;
        temp_d += temp * temp;
        filter_index += num_threads;
        image_index += num_threads;
    }
    
    // let staging_filter_index = offset_image_id * (num_filters * num_threads) + (wid.x * num_threads + lid.x);
    // var staging_filter_start = offset_image_id * (num_filters * num_threads) + (wid.x * num_threads);

    let staging_filter_index = lid.x;
    d_staging[staging_filter_index] = temp_d;
    staging_filter[staging_filter_index] = temp_re_d;    

    // sync threads
    workgroupBarrier();

    // get size of workgroup 
    var size = num_threads/2u;
    
    // sums up D * Re in staging_filter
    while (size != 0) {
        if (staging_filter_index < size) {
            staging_filter[staging_filter_index] += staging_filter[staging_filter_index + size];    
            d_staging[staging_filter_index] = d_staging[staging_filter_index] + d_staging[staging_filter_index + size];    
        }

        // if (index_d_filter < index_d_filter_start + size) {
        // }

        workgroupBarrier();
        size = size / 2;
    }

      
    // makes the cosine similarity in staging_filter with spaces of 256
    if (lid.x == 0) {
        let off_set = offset_image_id * num_filters;
        let value = staging_filter[staging_filter_index]/ sqrt(d_staging[staging_filter_index]); /// sqrt(d_staging[index_d_filter]);
        out[wid.x + off_set] = value; //sqrt(d_staging[staging_filter]);  
    }
}    