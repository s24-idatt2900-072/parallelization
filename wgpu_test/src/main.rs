use wgpu::util::DeviceExt;

fn main() {
    // TODO: initiate logger instead of print
    // data for input to the shader
    let a = vec![1, 2, 3]; // er int fordi floats parses feil
    let b = vec![1, 1, 1]; // er int fordi floats parses feil

    // create device and queue
    let d = WgpuDevice::new().expect("Failed to create device");
    println!("Choosen device: {:?}", d.dev);
    // create shader module
    let cs_module = d.create_shader_module(include_str!("dot_product.wgsl"));
    // memory size for the data
    // deler på to fordi the er int og ikke float
    let size = (std::mem::size_of_val(&a)/2) as wgpu::BufferAddress;
    println!("Size of data: {}", size);

    // Instantiates buffer with data as read_only.
    let storage_buffer_a = d.create_buffer_init(
        &a,
        wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
    );
    let storage_buffer_b = d.create_buffer_init(
        &b,
        wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
    );
    // Instantiates buffer for output.
    let storage_buffer_output = d.create_buffer(
        size, 
        Some("Output buffer"), 
            wgpu::BufferUsages::STORAGE
            //| wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
    );

    // Defines the bind group layout.
    let bind_group = d.create_bind_group( vec![&storage_buffer_a, &storage_buffer_b, &storage_buffer_output], &d.create_bind_group_layout(3, vec![true, true, false]));
    // Instantiates the pipeline.
    let pipeline_layout = d.dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&d.create_bind_group_layout(3, vec![true, true, false])],
        push_constant_ranges: &[],
    });
    let compute_pipeline = d.create_compute_pipeline(&cs_module, &pipeline_layout);

    // Creates the command encoder.
    let encoder =
        d.create_command_encoder(&compute_pipeline, &bind_group, a.len() );
    
    // Submits command encoder for processing
    d.submit(encoder);
        
    let mut res = vec![0; a.len()];
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(get_data(
        &mut res, 
        &storage_buffer_output,
        &d.dev,
        &d.que,));
    println!("res: {:?}", res);
    println!("Finished");
}

async fn get_data<T: bytemuck::Pod>(
    output: &mut [T],
    storage_buffer: &wgpu::Buffer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: storage_buffer.size(), 
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    println!("staging_buffer: {:?}", staging_buffer.size());
    println!("storage_buffer: {:?}", storage_buffer.size());

    command_encoder.copy_buffer_to_buffer(
        storage_buffer,
        0,
        &staging_buffer,
        0,
        storage_buffer.size(),
    );
    queue.submit(Some(command_encoder.finish()));
    let buffer_slice = /*storage_buffer.slice(..);*/staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();
    
    receiver.recv_async().await.unwrap().unwrap();
    buffer_slice.get_mapped_range().iter().for_each(|x| print!("{} ", x));
    println!();
    output.copy_from_slice(bytemuck::cast_slice(&buffer_slice.get_mapped_range()[..]));
    staging_buffer.unmap();
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

    fn create_buffer(&self, size: u64, l: Option<&str>, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let label = match l {
            Some(l) => Some(l),
            None => Some("Data buffer"),
        };

        self.dev.create_buffer(&wgpu::BufferDescriptor{
            mapped_at_creation: false,
            label,
            size: size,
            usage,
        })
    }

    fn create_shader_module(&self, shader: &str) -> wgpu::ShaderModule {
        self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
        })
    }

    fn create_buffer_init<T>(&self, content: &Vec<T>, usage: wgpu::BufferUsages) -> wgpu::Buffer
    where
        T: bytemuck::Pod,
    {
        self.dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&content),
            usage,
        })
    }

    fn create_compute_pipeline(&self, shader: &wgpu::ShaderModule, layout: &wgpu::PipelineLayout) -> wgpu::ComputePipeline {
        self.dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module: &shader,
            entry_point: "main",
        })
    }

    fn create_bind_group_layout(&self, binds: usize, read_only: Vec<bool>) -> wgpu::BindGroupLayout {
        let mut entries = Vec::new();
        for i in 0..binds {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: read_only[i]},
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
        }
        self.dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        })
    }

    fn create_bind_group(&self, buffers: Vec<&wgpu::Buffer>, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup{
        let mut entries = Vec::new();
        for (i, b) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
        }        
        self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &entries,
        })
    }

    fn create_command_encoder(&self, comp_pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, cells: usize) -> wgpu::CommandEncoder {
        self.dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut encoder = self.dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(comp_pipe);
            cpass.set_bind_group(0, bind, &[]);
            cpass.insert_debug_marker("compute shader");
            cpass.dispatch_workgroups(cells as u32, 1, 1);
        }
        encoder
    }

    fn submit(&self, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.que.submit(Some(enc.finish()))
    }
}
