use wgpu::util::DeviceExt;

pub struct WgpuContext {
    dev: wgpu::Device,
    que: wgpu::Queue,
}

impl WgpuContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let rt = tokio::runtime::Runtime::new()?;
        let (dev, que) = rt.block_on(Self::get_device())?;
        Ok(Self { dev, que })
    }

    pub fn compute_gpu<T>(
        &self,
        shader: &str,
        buffers: &mut Vec<&wgpu::Buffer>,
        dis_size: (u32, u32, u32),
        writers: usize,
    ) where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        // Defines the bind group layout.
        let layout = self.bind_layout(buffers.len(), writers);
        // Instantiates the bind group.
        let bind_group = self.bind_group(buffers, &layout);
        // Create shader module.
        let shader = self.shader_mod(shader);
        // Instantiates the pipeline.
        let compute_pipeline = self.pipeline(&shader, &layout);
        // Creates the command encoder.
        let encoder = self.command_enc(&compute_pipeline, &bind_group, dis_size);
        // Submits command encoder for processing.
        self.submit(encoder);
    }

    async fn get_device() -> Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError> {
        let adapter = wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("No matching adapter for preferences");
        let limits = adapter.limits();
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await
    }

    pub fn shader_mod(&self, shader: &str) -> wgpu::ShaderModule {
        self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
        })
    }

    pub fn read_write_buf(&self, size: u64) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Output buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn read_only_buf<T>(&self, content: &Vec<T>) -> wgpu::Buffer
    where
        T: bytemuck::Pod,
    {
        self.dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&content),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    pub fn staging_buf(&self, size: u64) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Staging buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        })
    }

    pub fn pipeline(
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

    pub fn bind_layout(&self, binds: usize, writers: usize) -> wgpu::BindGroupLayout {
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
                                        read_only: i < binds - writers,
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

    pub fn bind_group(
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

    pub fn command_enc(
        &self,
        comp_pipe: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        size: (u32, u32, u32),
    ) -> wgpu::CommandEncoder {
        let (x, y, z) = size;
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
            cpass.dispatch_workgroups(x, y, z);
        }
        enc
    }

    pub fn submit(&self, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.que.submit(Some(enc.finish()))
    }

    pub fn get_data<T: bytemuck::Pod + std::fmt::Debug>(
        &self,
        output: &mut Vec<Vec<T>>,
        out_buf: &wgpu::Buffer,
    ) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let _ = rt.block_on(self.get_data_async(output, &out_buf));
    }

    async fn get_data_async<T: bytemuck::Pod + std::fmt::Debug>(
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
        enc.copy_buffer_to_buffer(storage_buf, 0, &staging_buf, 0, storage_buf.size());
        self.submit(enc);

        // Defines slice for data transfer and waits for signal
        let buf_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buf_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.dev.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Receives signal and copies data over
        let _ = receiver.recv_async().await?;
        let flat_output = Vec::from(bytemuck::cast_slice(&buf_slice.get_mapped_range()[..]));
        println!("flatt output: {:?}", flat_output);
        let mut it = flat_output.into_iter();
        let _ = output
            .iter_mut()
            .map(|inner| inner.iter_mut().for_each(|r| *r = it.next().unwrap()))
            .collect::<Vec<_>>();

        staging_buf.unmap();
        Ok(())
    }
}
