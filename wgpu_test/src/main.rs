#[tokio::main]
async fn main() -> std::io::Result<()>{
    // TODO: initiate logger instead of print
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
    let d = WgpuDevice::new().expect("Failed to create device");
    println!("Choosen device: {:?}", d.dev);
    let b = d.create_buffer(20);
    d.que.write_buffer(&b, 0, &[0x01; 20]);
    println!("Finished");
    Ok(())
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

    pub fn create_buffer(&self, size: usize) -> wgpu::Buffer {
        self.dev.create_buffer(&wgpu::BufferDescriptor{
            mapped_at_creation: false,
            label: Some("data buffer"),
            size: Self::byte_size(size),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn byte_size(size: usize) -> wgpu::BufferAddress {
        u64::try_from(size * std::mem::size_of::<u64>()).unwrap()
    }
}
