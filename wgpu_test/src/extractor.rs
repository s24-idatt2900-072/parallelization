use crate::wgpu_context::WgpuContext;
use crate::wgpu_context_error::WgpuContextError;

/// Extractor performs feature extraction using WGPU operations.
pub struct Extractor {
    con: WgpuContext,
}

impl Extractor {
    /// Creates a new instance of the Extractor with a WGPU context.
    pub fn new() -> Result<Self, WgpuContextError> {
        let con = WgpuContext::new()?;
        Ok(Self { con })
    }

    /// Performs the actual feature extraction using the provided data and parameters.
    ///
    /// # Arguments
    ///
    /// * `a` - Input data matrix A.
    /// * `b` - Input data matrix B.
    /// * `out` - Output matrix for storing computed features.
    /// * `chunk` - Size of the computation chunk.
    /// * `filter_chunk` - Size of the filter chunk.
    pub fn get_features<T>(
        &self,
        a: &Vec<Vec<T>>,
        b: &Vec<Vec<T>>,
        chunk: usize,
        filter_chunk: usize,
    ) -> Result<Vec<T>, WgpuContextError>
    where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        let inner_size = a.get(0).expect("No input provided").len();
        if b.iter().all(|i| i.len() != inner_size) {
            return Err(WgpuContextError::CustomWgpuContextError(String::from(
                "Can't compute with different lengths",
            )));
        }
        let max = self.con.get_limits().max_buffer_size as f64;
        println!("max buffer size: {}", max);
        // 325 images e min max med 100 000 filter
        let mut max_number_of_images = ((max
            / (std::mem::size_of::<T>() * inner_size * b.len()) as f64)
            * (chunk as f64 / (std::mem::size_of::<T>() * inner_size * b.len()) as f64))
            as usize;
        println!("MAX number of images: {}", max_number_of_images);
        // Memory size for the output data
        let mut size = ((b.len() * a.len() * inner_size * std::mem::size_of::<T>()) / 2)
            as wgpu::BufferAddress;
        if inner_size % 2 != 0 {
            size += ((a.len() * b.len() * std::mem::size_of::<T>()) / 2) as wgpu::BufferAddress;
            max_number_of_images = ((max
                / (std::mem::size_of::<T>() * (inner_size + 1) * b.len()) as f64)
                * (2 as f64 / (std::mem::size_of::<T>() * (inner_size + 1) * b.len()) as f64))
                as usize;
        }
        println!("MAX number of images: {}", max_number_of_images);
        // Instantiates buffers for computating.
        let buffers = [a, b]
            .iter()
            .map(|i| Self::flatten_content(i))
            .map(|i| self.con.storage_buf(&i).expect("Failed to create buffer"))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        println!("Inner size: {}", inner_size);
        println!("B size: {}", b.len());
        println!("A size: {}", a.len());
        println!("Size: {}", size);
        println!(
            "Size: {}",
            ((b.len() * a.len() * inner_size + a.len() * b.len()) * (std::mem::size_of::<T>() / 2))
        );
        println!("Chunk: {}", chunk);
        println!("Filter chunk: {}", filter_chunk);
        let info_buf = self.con.storage_buf::<u32>(&vec![
            inner_size as u32,
            b.len() as u32,
            a.len() as u32,
            chunk as u32,
            filter_chunk as u32,
        ])?;
        buffers.insert(0, &info_buf);
        let out_buf = self.con.read_write_buf(size)?;
        buffers.push(&out_buf);

        let dis_size = Extractor::get_dispatch_size(
            a.len() as i32,
            b.len() as i32,
            inner_size as i32,
            chunk as i32,
        );
        println!("Dispatch size: {:?}", dis_size);
        self.con.compute_gpu::<T>(
            include_str!("shaders/dot_mult_2x_reduce.wgsl"),
            &mut buffers,
            dis_size,
            1,
        )?;
        let new_out = self
            .con
            .read_write_buf((a.len() * b.len() * std::mem::size_of::<T>()) as u64)?;
        let mut buffers = vec![&info_buf, &out_buf, &new_out];
        self.con.compute_gpu::<T>(
            include_str!("shaders/sum_reduction.wgsl"),
            &mut buffers,
            (65_536, 1, 1), //tar 196 608 sum med chunk = 5 som betyr 3 summer per workgroup
            1,
        )?;
        println!("Reading data");
        Ok(self.con.get_data(&new_out)?)
    }

    /// Flattens a 2D matrix into a 1D vector.
    ///
    /// This method takes a 2D matrix (`content`) and flattens it into a 1D vector,
    /// ensuring that the original order of elements is preserved. The resulting vector
    /// can be useful for operations where a linear representation of the matrix is required.
    ///
    /// # Arguments
    ///
    /// * `content` - The 2D matrix to be flattened.
    ///
    /// # Returns
    ///
    /// Returns a new `Vec<T>` containing the flattened elements.
    fn flatten_content<T>(content: &Vec<Vec<T>>) -> Vec<T>
    where
        T: bytemuck::Pod,
    {
        content.iter().flatten().cloned().collect()
    }

    /// Calculates the dispatch size for a WGPU compute shader.
    ///
    /// This method calculates the dispatch size for a WGPU compute shader based on the
    /// dimensions of input matrices (`a_len` and `b_len`). The dispatch size is calculated
    /// in workgroups, taking into account a predefined workgroup size.
    ///
    /// # Arguments
    ///
    /// * `a_len` - The length of matrix A (input).
    /// * `b_len` - The length of matrix B (input).
    /// * `i_len` - The length of the inner dimension of the matrices.
    /// * `chunk` - The size of the computation chunk.
    ///
    /// # Returns
    ///
    /// Returns a tuple `(x, y, z)` representing the calculated dispatch size.
    fn get_dispatch_size(a_len: i32, b_len: i32, i_len: i32, chunk: i32) -> (u32, u32, u32) {
        let workgroup_size = 16;
        let out_len = a_len * b_len * i_len;
        let x = (out_len / chunk) / workgroup_size;
        let y = b_len / workgroup_size;
        let x = if out_len.rem_euclid(workgroup_size) != 0 {
            x + 1
        } else {
            x
        };
        let y = if b_len.rem_euclid(workgroup_size) != 0 {
            y + 1
        } else {
            y
        };
        //(x as u32, y as u32, 1)
        (300, 7_000, 1)
    }
}
