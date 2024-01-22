fn main() {
    // TODO: initiate logger instead of print
    let instance = wgpu::Instance::default();
    println!("Available backends:");
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        println!("{:?}", a.get_info());
    }

    let d = WgpuDevice::new().expect("Failed to create device");
    let b = d.create_buffer(20, wgpu::BufferUsages::STORAGE, None);
    d.que.write_buffer(&b, 0, &[0x01; 20]);
    println!("Finished");
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
            .expect("No adapter matching preferences")
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

    pub fn create_buffer(&self, size: usize, usage: wgpu::BufferUsages, label: Option<&str>) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor{
            size: byte_size(size) as wgpu::BufferAddress,
            usage,
            mapped_at_creation: false,
            label,
        })
    }
}

fn byte_size(size: usize) -> u64 {
    u64::try_from(size * std::mem::size_of::<u64>()).unwrap()
}
