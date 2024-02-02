use wgpu::util::DeviceExt;

fn main() {
    // TODO: initiate logger instead of print
    //print_devices();
    // Data for computation
    let a: Vec<Vec<f32>> = vec![vec![1., 2., 3.], vec![2., 4., 6.]];
    let b: Vec<Vec<f32>> = vec![vec![3., 2., 1.], vec![3., 3., 3.]];
    // Result buffer
    let mut res: Vec<Vec<f32>> = vec![vec![0.; b.len()]; a.len()];

    WgpuDevice::dot(&a, &b, &mut res);
    println!("Result dot: {:?}", res);
    //WgpuDevice::max(&a, &mut res);
    //println!("Result max: {:?}", res);
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
        println!(
            "{:?}",
            a.request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None
            )
            .await
            .unwrap()
            .0
            .global_id()
        );
    }
}
pub struct WgpuDevice {
    dev: wgpu::Device,
    que: wgpu::Queue,
}

impl WgpuDevice {
    pub fn dot<T>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>, out: &mut Vec<Vec<T>>)
    where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        WgpuDevice::new().unwrap().dot_product(a, b, out);
    }

    pub fn max<T>(a: &Vec<T>, out: &mut Vec<T>)
    where
        T: bytemuck::Pod,
    {
        WgpuDevice::new().unwrap().max_pool(a, out);
    }

    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let rt = tokio::runtime::Runtime::new()?;
        let (dev, que) = rt.block_on(Self::get_device())?;
        Ok(Self { dev, que })
    }

    pub fn dot_product<T>(&self, a: &Vec<Vec<T>>, b: &Vec<Vec<T>>, out: &mut Vec<Vec<T>>)
    where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        let inner_size = a.get(0).expect("No input provided").len();
        if b.iter().all(|i| i.len() != inner_size) {
            panic!("Can't compute with different lengths");
        }
        // Memory size for the output data
        let size = (std::mem::size_of::<T>() * b.len() * a.len()) as wgpu::BufferAddress;
        println!("size: {:?}", size);
        // Instantiates buffers for computating.
        let buffers = [a, b]
            .iter()
            .map(|i| self.flatten_content(i))
            .map(|i| self.read_only_buf(&i))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        println!("Inner and outer {} / {}", inner_size, a.len());
        let info_buf = self.read_only_buf::<u32>(&vec![inner_size as u32, a.len() as u32]);
        buffers.push(&info_buf);
        let out_buf = self.read_write_buf(size);
        buffers.push(&out_buf);

        self.compute_method(
            include_str!("dot_product.wgsl"),
            &mut buffers,
            &out_buf,
            out,
            size,
        );
    }

    pub fn max_pool<T>(&self, a: &Vec<T>, out: &mut Vec<T>)
    where
        T: bytemuck::Pod,
    {
        todo!();
        //self.compute_method(include_str!("max_pool.wgsl"), &[a], out);
    }

    fn compute_method<T>(
        &self,
        shader: &str,
        buffers: &mut Vec<&wgpu::Buffer>,
        out_buf: &wgpu::Buffer,
        out: &mut Vec<Vec<T>>,
        size: u64,
    ) where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        // Defines the bind group layout.
        let layout = self.bind_layout(buffers.len());
        // Instantiates the bind group.
        let bind_group = self.bind_group(buffers, &layout);
        // Create shader module.
        let shader = self.shader_mod(shader);
        // Instantiates the pipeline.
        let compute_pipeline = self.pipeline(&shader, &layout);
        // Creates the command encoder.
        let encoder = self.command_enc(&compute_pipeline, &bind_group);
        // Submits command encoder for processing.
        self.submit(encoder);

        let rt = tokio::runtime::Runtime::new().unwrap();
        println!("out_buf: {:?}", out_buf.size());
        let _ = rt.block_on(self.get_data(out, &out_buf));
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
                None,
            )
            .await
    }

    fn shader_mod(&self, shader: &str) -> wgpu::ShaderModule {
        self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
        })
    }

    fn read_write_buf(&self, size: u64) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Output buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn flatten_content<T>(&self, content: &Vec<Vec<T>>) -> Vec<T>
    where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        let flat_content: Vec<T> = content.iter().flatten().cloned().collect();
        println!("flat_content: {:?}", flat_content);
        flat_content
    }

    fn read_only_buf<T>(&self, content: &Vec<T>) -> wgpu::Buffer
    where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        self.dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&content),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn staging_buf(&self, size: u64) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Staging buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        })
    }

    fn pipeline(
        &self,
        shader: &wgpu::ShaderModule,
        bind_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let layout = &self
            .dev
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_layout],
                push_constant_ranges: &[],
            });
        self.dev
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(layout),
                module: &shader,
                entry_point: "main",
            })
    }

    fn bind_layout(&self, binds: usize) -> wgpu::BindGroupLayout {
        self.dev
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[0..binds]
                    .into_iter()
                    .map(|r| {
                        r.into_iter()
                            .map(|i| wgpu::BindGroupLayoutEntry {
                                binding: i as u32,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage {
                                        read_only: i != binds - 1,
                                    },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            })
                            .collect::<Vec<wgpu::BindGroupLayoutEntry>>()
                    })
                    .collect::<Vec<Vec<wgpu::BindGroupLayoutEntry>>>()
                    .get(0)
                    .expect("Failed to create layout"),
            })
    }

    fn bind_group(
        &self,
        buffers: &mut Vec<&wgpu::Buffer>,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
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

    fn command_enc(
        &self,
        comp_pipe: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
    ) -> wgpu::CommandEncoder {
        let max_dispatch = 65_535;
        let mut enc = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(comp_pipe);
            cpass.set_bind_group(0, bind, &[]);
            cpass.insert_debug_marker("compute shader");
            cpass.dispatch_workgroups(1, 1, 1);
        }
        enc
    }

    fn submit(&self, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.que.submit(Some(enc.finish()))
    }

    async fn get_data<T: bytemuck::Pod + std::fmt::Debug>(
        &self,
        output: &mut Vec<Vec<T>>,
        storage_buf: &wgpu::Buffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut enc = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Data transfer"),
            });
        // Creates staging buffer for data transfer
        let staging_buf = self.staging_buf(storage_buf.size());
        println!("staging_buf: {:?}", staging_buf.size());
        enc.copy_buffer_to_buffer(storage_buf, 0, &staging_buf, 0, storage_buf.size());
        self.submit(enc);

        // Defines slice for data transfer and waits for signal
        let buf_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buf_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.dev.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Receives signal and copies data over
        let _ = receiver.recv_async().await?;
        //output.copy_from_slice(bytemuck::cast_slice(&buf_slice.get_mapped_range()[..]));
        let flat_output = Vec::from(bytemuck::cast_slice(&buf_slice.get_mapped_range()[..]));
        println!("flat_output: {:?}", flat_output);

        let mut it = flat_output.into_iter();
        let _ = output
            .iter_mut()
            .map(|inner| inner.iter_mut().for_each(|r| *r = it.next().unwrap()))
            .collect::<Vec<_>>();

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
    //WgpuDevice::dot(&a, &b, &mut res);
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
