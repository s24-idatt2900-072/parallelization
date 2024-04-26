use crate::wgpu_context::WgpuContext;
use crate::wgpu_context_error::WgpuContextError;
use rayon::prelude::*;

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

    /// Computes the dot product of two matrices using WGPU.
    ///
    /// This method computes the dot product of two matrices `a` and `b` using WGPU. The matrices
    /// are passed as 2D vectors, and the result is returned as a 1D vector. The `dis` parameter
    /// specifies the number of workgroups to use in the computation.
    ///
    /// # Arguments
    ///
    /// * `a` - The first matrix.
    /// * `b` - The second matrix.
    /// * `dis` - The number of workgroups to use in the computation.
    /// * `shader` - The shader to use for the computation. The default shader is `parallel_dot.wgsl`.
    ///
    /// # Returns
    ///
    /// Returns a new `Vec<T>` containing the result of the dot product.
    pub fn dot<T>(
        &self,
        a: &Vec<Vec<T>>,
        b: &Vec<Vec<T>>,
        dis: (u32, u32, u32),
        shader: &str,
    ) -> Result<Vec<T>, WgpuContextError>
    where
        T: bytemuck::Pod,
    {
        let size = (a.len() * b.len() * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let buffers = [a, b]
            .iter()
            .map(|i| Self::flatten_content(i))
            .map(|i| self.con.storage_buf(&i).expect("Failed to create buffer"))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        let out_buf = self.con.read_write_buf(size)?;
        buffers.push(&out_buf);

        let start = std::time::Instant::now();
        let sub_in = self.con.compute_gpu::<T>(shader, &mut buffers, dis, 1)?;
        self.con.poll_execution(sub_in);
        println!("Elapsed time shader computation: {:?}", start.elapsed());
        self.con.get_data::<T>(&out_buf)
    }

    /// Computes the cosine similarity and max pooling of two matrices using WGPU.
    ///
    /// # Arguments
    ///
    /// * `image` - The image matrix.
    /// * `re` - The real part of the filter matrix.
    /// * `abs` - The absolute part of the filter matrix.
    /// * `cosine_dis` - The number of workgroups to use in the cosine similarity computation.
    /// * `max_dis` - The number of workgroups to use in the max pooling computation.
    /// * `cosine_shader` - The shader to use for the cosine similarity computation.
    /// * `max_shader` - The shader to use for the max pooling computation.
    /// * `out_len` - The length of the output vector.
    /// * `max_chunk` - The maximum chunk size for the max pooling computation.
    ///
    /// # Returns
    ///
    /// Returns a new `Vec<T>` containing the result of the computation.
    pub fn compute_cosine_simularity_max_pool_all_images<T>(
        &self,
        image: &Vec<T>,
        re: &Vec<T>,
        abs: &Vec<T>,
        cosine_dis: (u32, u32, u32),
        max_dis: (u32, u32, u32),
        cosine_shader: &str,
        max_shader: &str,
        out_len: usize,
        max_chunk: u64,
    ) -> Result<Vec<T>, WgpuContextError>
    where
        T: bytemuck::Pod,
    {
        let size = (out_len * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        let buffers = [image, re, abs]
            .iter()
            .map(|i| self.con.storage_buf(&i).expect("Failed to create buffer"))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        let out_buf = self.con.read_write_buf(size)?;
        buffers.push(&out_buf);
        let _ = self
            .con
            .compute_gpu::<T>(cosine_shader, &mut buffers, cosine_dis, 1)?;

        let mut buffers = vec![&out_buf];
        let max_out_buf = self
            .con
            .read_write_buf((size + max_chunk - 1) / max_chunk)?;
        buffers.push(&max_out_buf);
        let _ = self
            .con
            .compute_gpu::<T>(max_shader, &mut buffers, max_dis, 1)?;
        self.con.get_data::<T>(&max_out_buf)
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
}

/// Tests the result of a computation.
pub fn test_res(res: Vec<f32>, expected_res: f32) {
    println!("Finalizing temp buffer dot shader..");
    let total_elements = res.len();
    let wrong_elements = res.par_iter().filter(|i| **i != expected_res).count();
    let percentage_wrong = (wrong_elements as f64 / total_elements as f64) * 100.0;
    println!("Total number of elements: {}", total_elements);
    println!("Number of elements wrong: {}", wrong_elements);
    println!("Percentage of elements wrong: {:.2}%", percentage_wrong);
}
