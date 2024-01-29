use wgpu::util::DeviceExt;

fn main() {
    // TODO: initiate logger instead of print

    // Data for computation
    let a: Vec<f32> = vec![1., 2., 3.]; 
    let b: Vec<f32> = vec![3., 2., 1.]; 
    // Result buffer
    let mut res: Vec<f32> = vec![0.; a.len()];

    WgpuDevice::dot(&a, &b, &mut res);
    println!("Result dot: {:?}", res);
    WgpuDevice::max(&a, &mut res);
    println!("Result max: {:?}", res);
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
    pub fn dot<T>(a: &Vec<T>, b: &Vec<T>, out: &mut Vec<T>) 
    where T: bytemuck::Pod {
        WgpuDevice::new().unwrap().dot_product(a, b, out);
    }

    pub fn max<T>(a: &Vec<T>, out: &mut Vec<T>) 
    where T: bytemuck::Pod {
        WgpuDevice::new().unwrap().max_pool(a, out);
    }

    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let rt = tokio::runtime::Runtime::new()?;
        let (dev, que) = rt.block_on(Self::get_device())?;
        Ok(Self { dev, que })
    }

    // Flow of computation
    // Shader -> shaderfile
    // Buffers -> usage & size -> Some(data)
    // B-layout -> read_only[bool]
    // Bind group -> buffers[] & b-layout
    // Pipe -> b-layout for p-layout & shader
    // Encoder -> pipeline & bind group
    // Submit encoder to compute shader
    pub fn dot_product<T>(&self, a: &Vec<T>, b: &Vec<T>, out: &mut Vec<T>) 
        where T: bytemuck::Pod {
        // Memory size for the data
        let size = (std::mem::size_of::<f32>()*a.len()) as wgpu::BufferAddress;
        // Instantiates buffers for computating.
        let buf_a = self.read_only_buf(&a);
        let buf_b = self.read_only_buf(&b);
        let out_buf = self.output_buf(size);
        // Defines the bind group layout.
        let layout = self.bind_layout(vec![true, true, false]);
        // Instantiates the bind group.
        let bind_group = self.bind_group( vec![&buf_a, &buf_b, &out_buf], &layout);
        // Create shader module.
        let shader = self.shader_mod(include_str!("dot_product.wgsl"));
        // Instantiates the pipeline.
        let compute_pipeline = self.pipeline(&shader, &layout);
        // Creates the command encoder.
        let encoder =
        self.command_enc(&compute_pipeline, &bind_group, a.len() as u32);
        // Submits command encoder for processing.
        self.submit(encoder);
            
        // Consumes async trait and copies data
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _ = rt.block_on(self.get_data(
            out, 
            &out_buf));
    }

    pub fn max_pool<T>(&self, a: &Vec<T>, out: &mut Vec<T>) 
        where T: bytemuck::Pod {
        // Memory size for the data
        let size = (std::mem::size_of::<f32>()*a.len()) as wgpu::BufferAddress;
        // Instantiates buffers for computating.
        let buf_a = self.read_only_buf(&a);
        let out_buf = self.output_buf(size);
        // Defines the bind group layout.
        let layout = self.bind_layout(vec![true, false]);
        // Instantiates the bind group.
        let bind_group = self.bind_group( vec![&buf_a, &out_buf], &layout);
        // Create shader module.
        let shader = self.shader_mod(include_str!("max.wgsl"));
        // Instantiates the pipeline.
        let compute_pipeline = self.pipeline(&shader, &layout);
        // Creates the command encoder.
        let encoder =
        self.command_enc(&compute_pipeline, &bind_group, a.len() as u32);
        // Submits command encoder for processing.
        self.submit(encoder);
            
        // Consumes async trait and copies data
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _ = rt.block_on(self.get_data(
            out, 
            &out_buf));
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
    
    fn shader_mod(&self, shader: &str) -> wgpu::ShaderModule {
        self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
        })
    }

    fn output_buf(&self, size: u64) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor{
            mapped_at_creation: false,
            label: Some("Output buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn read_only_buf<T>(&self, content: &Vec<T>) -> wgpu::Buffer
    where
        T: bytemuck::Pod,
    {
        self.dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&content),
            usage: wgpu::BufferUsages::STORAGE
        })
    }

    fn staging_buf(&self, size: u64) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor{
            mapped_at_creation: false,
            label: Some("Staging buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        })
    }

    fn pipeline(&self, shader: &wgpu::ShaderModule, bind_layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
        let layout = &self.dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        self.dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module: &shader,
            entry_point: "main",
        })
    }

    fn bind_layout(&self, read_only: Vec<bool>) -> wgpu::BindGroupLayout {
        self.dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &read_only
                .into_iter()
                .enumerate()
                .map(|(i, read_only)| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }).collect::<Vec<wgpu::BindGroupLayoutEntry>>(),
        })
    }

    fn bind_group(&self, buffers: Vec<&wgpu::Buffer>, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {   
        self.dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<wgpu::BindGroupEntry>>(),
        })
    }

    fn command_enc(&self, comp_pipe: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, cells: u32) -> wgpu::CommandEncoder {
        let mut enc = self.dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(comp_pipe);
            cpass.set_bind_group(0, bind, &[]);
            cpass.insert_debug_marker("compute shader");
            cpass.dispatch_workgroups(cells, 1, 1);
        }
        enc
    }

    fn submit(&self, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.que.submit(Some(enc.finish()))
    }

    async fn get_data<T: bytemuck::Pod>(
        &self, 
        output: &mut [T],
        storage_buf: &wgpu::Buffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut enc = self
            .dev
            .create_command_encoder(
                &wgpu::CommandEncoderDescriptor { 
                    label: Some("Data transfer") 
                }
            );
        // Creates staging buffer for data transfer
        let staging_buf = self.staging_buf(storage_buf.size());
        enc.copy_buffer_to_buffer(
            storage_buf,
            0,
            &staging_buf,
            0,
            storage_buf.size(),
        );
        self.submit(enc);

        // Defines slice for data transfer and waits for signal
        let buf_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buf_slice.map_async(
            wgpu::MapMode::Read, 
            move |r| sender.send(r).unwrap()
        );
        self.dev.poll(wgpu::Maintain::wait()).panic_on_timeout();
        
        // Receives signal and copies data over
        let _ = receiver.recv_async().await?;
        output.copy_from_slice(bytemuck::cast_slice(&buf_slice.get_mapped_range()[..]));
        staging_buf.unmap();
        Ok(())
    }
}

#[test]
fn test_dot() {
    // Data for computation
    let a: Vec<f32> = vec![1., 2., 3.]; 
    let b: Vec<f32> = vec![3., 2., 1.]; 
    // Result buffer
    let mut res: Vec<f32> = vec![0.; a.len()];
    WgpuDevice::dot(&a, &b, &mut res);
    assert_eq!(res[0], 10.);
}

#[test]
fn test_max() {
    // Data for computation
    let a: Vec<f32> = vec![1., 2., 3.]; 
    // Result buffer
    let mut res: Vec<f32> = vec![0.; a.len()];
    WgpuDevice::max(&a, &mut res);
    assert_eq!(res[0], 3.);
}