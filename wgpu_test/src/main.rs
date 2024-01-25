use wgpu::util::DeviceExt;
//#[tokio::main]
/*async*/ fn main() -> std::io::Result<()>{
    // TODO: initiate logger instead of print
    //print_devices();
    //println!();
    
    // data for input to the shader
    let numbers = vec![1, 2, 3, 4, 5];

    // create device and queue
    let d = WgpuDevice::new().expect("Failed to create device");
    println!("Choosen device: {:?}", d.dev);

    // create shader module
    let cs_module = d.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("dot_shader.wgsl"))),
    });

    // memory size for the data
    let size = std::mem::size_of_val(&numbers) as wgpu::BufferAddress;
    let size = 20;
    println!("Size of data: {}", size);

    // Instantiates buffer without data.
    // `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    // `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let b = d.create_buffer(size, None);
    println!("Buffer created {}", b.size());

    // Instantiates buffer with data (`numbers`).
    let storage_buffer = d.dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&numbers),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    });
    println!("Storage buffer created {}", storage_buffer.size());

    // Instantiates the pipeline.
    let compute_pipeline = d.dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    // Defines the bind group layout.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = d.dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    // Creates the command encoder.
    let mut encoder =
        d.dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch_workgroups(numbers.len() as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
    }

    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &b, 0, size);

    // Submits command encoder for processing
    d.que.submit(Some(encoder.finish()));

    // Creates a buffer slice from the staging buffer.
    let buffer_slice = b.slice(..);
    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Waits for the result of the mapping.
    d.dev.poll(wgpu::Maintain::wait()).panic_on_timeout();

    let rt = tokio::runtime::Runtime::new().unwrap();
    if let Ok(Ok(())) = rt.block_on(receiver.recv_async()) {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Since contents are got in bytes, this converts these bytes back to u32
        let result: Vec<u32> = bytemuck::cast_slice(&data).to_vec();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        b.unmap(); // Unmaps buffer from memory, effectively freeing memory

        // Returns data from buffer
        let disp_steps: Vec<String> = result
        .iter()
        .map(|&n| match n {
            0xffffffff => "OVERFLOW".to_string(),
            _ => n.to_string(),
        })
        .collect();

    println!("Steps: [{}]", disp_steps.join(", "));
    } else {
        panic!("failed to run compute on gpu!")
    }

    println!("Finished");
    Ok(())
}

fn print_devices() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(print_devices_async());
}

async fn print_devices_async() {
    let instance = wgpu::Instance::default();
    println!("Available backends:");
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        println!("{:?}", a.get_info());
        println!("{:?}",a.request_device(
            &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None
        ).await.unwrap().0.global_id());
    }
}
pub struct WgpuDevice {
    dev: wgpu::Device,
    que: wgpu::Queue,
}

impl WgpuDevice {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let rt = tokio::runtime::Runtime::new()?;
        let (dev, que) = rt.block_on(Self::get_device())?;
        Ok(Self { dev, que })
    }

    async fn get_device() -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
        wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptionsBase { 
                power_preference: wgpu::PowerPreference::HighPerformance, 
                force_fallback_adapter: false, 
                compatible_surface: None, 
            })
            .await
            .expect("No matching adapter for preferences")
            .request_device(
                &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None
            )
            .await
    }

    fn create_buffer(&self, size: u64, l: Option<&str>) -> wgpu::Buffer {
        let label = match l {
            Some(l) => Some(l),
            None => Some("Data buffer"),
        };

        self.dev.create_buffer(&wgpu::BufferDescriptor{
            mapped_at_creation: false,
            label,
            size: size,
            usage: //wgpu::BufferUsages::STORAGE
                /*|*/ wgpu::BufferUsages::COPY_DST
                //| wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::MAP_READ,
        })
    }
}
