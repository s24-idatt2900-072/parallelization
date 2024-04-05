use crate::wgpu_context_error::WgpuContextError;
use wgpu::util::DeviceExt;

/// Represents a context for WGPU operations, including a WGPU device and queue.
///
/// The `WgpuContext` struct encapsulates a WGPU device and queue, providing a context for
/// performing various GPU operations.
pub struct WgpuContext {
    dev: wgpu::Device,
    que: wgpu::Queue,
}

impl WgpuContext {
    /// Creates a new instance of the `MyWgpuContext`.
    ///
    /// This function initializes a new instance of the `MyWgpuContext` by creating a runtime, obtaining
    /// a WGPU device and queue, and returning a result indicating success or an error.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized `WgpuContext` or a `WgpuContextError`
    pub fn new() -> Result<Self, WgpuContextError> {
        let rt = tokio::runtime::Runtime::new()?;
        let (dev, que) = rt.block_on(Self::get_device())?;
        Ok(Self { dev, que })
    }

    /// Executes a compute shader on the GPU using specified parameters.
    ///
    /// This function orchestrates the execution of a compute shader on the GPU with the provided
    /// shader source code (`shader`), a vector of GPU buffers (`buffers`), and parameters
    /// specifying the workgroup dimensions (`dis_size`) and the number of read-write buffers (`writers`).
    ///
    /// # Arguments
    ///
    /// * `shader` - The WGSL source code for the compute shader.
    /// * `buffers` - A mutable vector containing GPU buffers used as input and output.
    /// * `dis_size` - A tuple specifying the workgroup dimensions in (x, y, z) dimensions.
    /// * `writers` - The number of read-write buffers in the bind group.
    ///
    /// # Returns
    ///
    /// Returns a `Result` indicating the success or failure of the compute shader execution.
    pub fn compute_gpu<T>(
        &self,
        shader: &str,
        buffers: &mut Vec<&wgpu::Buffer>,
        dis_size: (u32, u32, u32),
        writers: usize,
    ) -> Result<(), WgpuContextError>
    where
        T: bytemuck::Pod,
    {
        // Defines the bind group layout.
        let layout = self.bind_layout(buffers.len(), writers)?;
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
        Ok(())
    }

    /// Asynchronously requests a WGPU device and queue based on specified preferences.
    ///
    /// This asynchronous function requests a WGPU device and queue from the default WGPU instance
    /// based on specified preferences such as power preference and device compatibility.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a tuple with the obtained `wgpu::Device` and `wgpu::Queue`,
    /// or a `wgpu::RequestDeviceError` if the request fails.
    async fn get_device() -> Result<(wgpu::Device, wgpu::Queue), WgpuContextError> {
        let adapter = wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptionsBase {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await;
        let adapter = match adapter {
            Some(adapter) => adapter,
            None => return Err(WgpuContextError::NoAdapterError),
        };
        let limits = adapter.limits();
        Ok(adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                },
                None,
            )
            .await?)
    }

    /// Creates a GPU shader module from a provided WGSL source code.
    ///
    /// This function generates a `wgpu::ShaderModule` using the provided WGSL source code (`shader`).
    ///
    /// # Arguments
    ///
    /// * `shader` - The WGSL source code as a string.
    ///
    /// # Returns
    ///
    /// Returns a `wgpu::ShaderModule` representing the compiled GPU shader.
    pub fn shader_mod(&self, shader: &str) -> wgpu::ShaderModule {
        self.dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader)),
        })
    }

    /// Creates a read-write GPU buffer with specified size.
    ///
    /// This function generates a `wgpu::Buffer` configured as a read-write storage buffer
    /// with the specified size (`size`).
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the read-write storage buffer in bytes.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `wgpu::Buffer` or a `WgpuContextError`.
    pub fn read_write_buf(&self, size: u64) -> Result<wgpu::Buffer, WgpuContextError> {
        self.check_limits(&(size as u32))?;
        Ok(self.dev.create_buffer(&wgpu::BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Read write buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        }))
    }

    /// Creates a read-only GPU buffer initialized with data from a vector.
    ///
    /// This function generates a `wgpu::Buffer` configured as a read-only storage buffer,
    /// and it initializes the buffer with the provided data from the vector (`content`).
    ///
    /// # Arguments
    ///
    /// * `content` - A vector containing the data to be stored in the GPU buffer.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `wgpu::Buffer` or a `WgpuContextError`.
    pub fn storage_buf<T>(&self, content: &Vec<T>) -> Result<wgpu::Buffer, WgpuContextError>
    where
        T: bytemuck::Pod,
    {
        let size = (content.len() * std::mem::size_of::<T>()) as u32;
        self.check_limits(&size)?;
        Ok(self
            .dev
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&content),
                usage: wgpu::BufferUsages::STORAGE,
            }))
    }

    /// Creates a staging buffer for data transfer between CPU and GPU.
    ///
    /// This function generates a `wgpu::Buffer` for use as a staging buffer, facilitating
    /// data transfer between the CPU and GPU. The buffer is labeled as "Staging buffer."
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the staging buffer in bytes.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `wgpu::Buffer` or a `WgpuContextError`.
    pub fn staging_buf(&self, size: u64) -> Result<wgpu::Buffer, WgpuContextError> {
        self.check_limits(&(size as u32))?;
        Ok(self.dev.create_buffer(&wgpu::BufferDescriptor {
            mapped_at_creation: false,
            label: Some("Staging buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        }))
    }

    /// Checks if the buffer size exceeds device limits.
    ///
    /// This function checks if the specified buffer size (`size`) exceeds the device limits.
    ///
    /// # Arguments
    ///
    /// * `size` - The size of the buffer to be checked.
    ///
    /// # Returns
    ///
    /// Returns a `Result` indicating the success or failure of the buffer size check.
    fn check_limits(&self, size: &u32) -> Result<(), WgpuContextError> {
        let limits = self.dev.limits();
        if size > &limits.max_storage_buffer_binding_size {
            return Err(WgpuContextError::ExceededBufferSizeError);
        }
        Ok(())
    }

    /// Creates a compute pipeline for a compute shader with a specified bind group layout.
    ///
    /// This function generates a `wgpu::ComputePipeline` for a compute shader using the provided
    /// shader module (`shader`) and bind group layout (`bind_layout`).
    ///
    /// # Arguments
    ///
    /// * `shader` - The `wgpu::ShaderModule` representing the compiled compute shader.
    /// * `bind_layout` - The `wgpu::BindGroupLayout` specifying the layout of the bind group for the shader.
    ///
    /// # Returns
    ///
    /// Returns a `wgpu::ComputePipeline` representing the computed compute pipeline.
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

    /// Creates a bind group layout for a compute shader with specified bindings.
    ///
    /// This function generates a `wgpu::BindGroupLayout` based on the number of
    /// bindings (`binds`) and the number of read-write bindings (`writers`).
    ///
    /// # Arguments
    ///
    /// * `binds` - The total number of bindings in the layout.
    /// * `writers` - The number of read-write bindings in the layout. Makes the last `writers` bindings as read-write.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the created `wgpu::BindGroupLayout` or a `WgpuContextError`.
    pub fn bind_layout(
        &self,
        binds: usize,
        writers: usize,
    ) -> Result<wgpu::BindGroupLayout, WgpuContextError> {
        if writers > binds {
            return Err(WgpuContextError::BindGroupError);
        }
        Ok(self
            .dev
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
                    .ok_or(WgpuContextError::BindGroupError)?,
            }))
    }

    /// Creates a bind group for a compute shader using specified buffers and layout.
    ///
    /// This function generates a `wgpu::BindGroup` based on the provided `buffers` and
    /// the specified `layout`. Each buffer is associated with a binding in the layout.
    ///
    /// # Arguments
    ///
    /// * `buffers` - A mutable vector of `wgpu::Buffer` references to be used in the bind group.
    /// * `layout` - The `wgpu::BindGroupLayout` specifying the layout of the bind group.
    ///
    /// # Returns
    ///
    /// Returns a `wgpu::BindGroup` representing the computed bind group.
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

    /// Creates a command encoder for a compute shader operation.
    ///
    /// This function generates a `wgpu::CommandEncoder` for executing a compute shader
    /// using the provided compute pipeline (`comp_pipe`), bind group (`bind`), and workgroup size (`size`).
    ///
    /// # Arguments
    ///
    /// * `comp_pipe` - The `wgpu::ComputePipeline` representing the compute shader to be executed.
    /// * `bind` - The `wgpu::BindGroup` containing the resources to be bound to the compute shader.
    /// * `dis_size` - A tuple representing the workgroup size in (x, y, z) dimensions.
    ///
    /// # Returns
    ///
    /// Returns a `wgpu::CommandEncoder` representing the encoded compute shader operation.
    pub fn command_enc(
        &self,
        comp_pipe: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        dis_size: (u32, u32, u32),
    ) -> wgpu::CommandEncoder {
        let (x, y, z) = dis_size;
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

    /// Submits a command encoder to the GPU queue for execution.
    ///
    /// This function submits a `wgpu::CommandEncoder` to the GPU queue, initiating the
    /// execution of the encoded commands.
    ///
    /// # Arguments
    ///
    /// * `enc` - The `wgpu::CommandEncoder` containing the commands to be submitted.
    ///
    /// # Returns
    ///
    /// Returns a `wgpu::SubmissionIndex` representing the index of the submitted commands.
    pub fn submit(&self, enc: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.que.submit(Some(enc.finish()))
    }

    /// Retrieves data from a GPU buffer and populates a vector.
    ///
    /// This function asynchronously retrieves data from the specified GPU buffer (`out_buf`)
    /// and populates the provided `output` vector. The data type `T` must implement the `bytemuck::Pod`
    /// trait
    ///
    /// # Arguments
    ///
    /// * `out_buf` - The `wgpu::Buffer` containing the data to be retrieved.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the populated vector or a `WgpuContextError`.
    pub fn get_data<T>(&self, out_buf: &wgpu::Buffer) -> Result<Vec<T>, WgpuContextError>
    where
        T: bytemuck::Pod,
    {
        let rt = tokio::runtime::Runtime::new().unwrap();
        Ok(rt.block_on(self.get_data_async(out_buf))?)
    }

    /// Asynchronously retrieves data from a GPU buffer and populates a vector.
    ///
    /// This asynchronous function retrieves data from the specified GPU buffer (`storage_buf`).
    /// The data type `T` must implement the `bytemuck::Pod` trait and be `Debug`.
    ///
    /// # Arguments
    ///
    /// * `output` - A mutable vector to store the retrieved data.
    /// * `storage_buf` - The `wgpu::Buffer` containing the data to be retrieved.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the populated vector or a `WgpuContextError`.
    async fn get_data_async<T>(
        &self,
        storage_buf: &wgpu::Buffer,
    ) -> Result<Vec<T>, WgpuContextError>
    where
        T: bytemuck::Pod,
    {
        let mut enc = self
            .dev
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Data transfer"),
            });
        // Creates staging buffer for data transfer
        let staging_buf = self.staging_buf(storage_buf.size())?;
        enc.copy_buffer_to_buffer(storage_buf, 0, &staging_buf, 0, storage_buf.size());
        self.submit(enc);

        // Defines slice for data transfer and waits for signal
        let buf_slice = staging_buf.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buf_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        self.dev.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Receives signal and copies data over
        let _ = receiver.recv_async().await?;
        let output = Vec::from(bytemuck::cast_slice(&buf_slice.get_mapped_range()[..]));
        staging_buf.unmap();
        Ok(output)
    }

    /// Retrieves the device limits for the WGPU context.
    ///
    /// This function retrieves the device limits for the WGPU context.
    ///
    /// # Returns
    ///
    /// Returns a `wgpu::Limits` struct containing the device limits.
    pub fn get_limits(&self) -> wgpu::Limits {
        self.dev.limits()
    }
}
