use crate::wgpu_context::WgpuContext;
use crate::wgpu_context_error::WgpuContextError;
use rayon::prelude::*;

pub fn test_res(res: Vec<f32>, alen: usize, blen: usize, expected_res: f32) {
    println!("Finalizing temp buffer dot shader..");
    let mut r = vec![vec![0.; blen]; alen];
    let chunk_sizes = r.iter().map(|inner| inner.len()).collect::<Vec<_>>();
    let mut starts = vec![0];
    let mut sum = 0;
    for size in &chunk_sizes {
        sum += *size;
        starts.push(sum);
    }
    r.par_iter_mut().enumerate().for_each(|(i, inner)| {
        let start = starts[i];
        let end = starts[i + 1];
        let slice = &res[start..end];
        inner
            .iter_mut()
            .enumerate()
            .for_each(|(j, r)| *r = slice[j]);
    });
    let total_elements = r.par_iter().flatten().count();
    let wrong_elements = r
        .par_iter()
        .flatten()
        .filter(|i| **i != expected_res)
        .count();
    let percentage_wrong = (wrong_elements as f64 / total_elements as f64) * 100.0;
    println!("Total number of elements: {}", total_elements);
    println!("Number of elements wrong: {}", wrong_elements);
    println!("Percentage of elements wrong: {:.2}%", percentage_wrong);
}

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
        T: std::fmt::Debug,
    {
        let max = self.con.get_limits().max_storage_buffer_binding_size;
        let max_images = (max as usize)
            .checked_div(std::mem::size_of::<T>() * b.len())
            .or(Some(1))
            .unwrap();
        println!("MAX number of images: {}", max_images);
        let size = (a.len() * b.len() * std::mem::size_of::<T>()) as wgpu::BufferAddress;

        let buffers = [a, b]
            .iter()
            .map(|i| Self::flatten_content(i))
            .map(|i| self.con.storage_buf(&i).expect("Failed to create buffer"))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        let out_buf = self.con.read_write_buf(size)?;
        let shader = match shader {
            "parallel_dot.wgsl" => include_str!("shaders/parallel_dot.wgsl"),
            "for_loop.wgsl" => include_str!("shaders/for_loop.wgsl"),
            _ => include_str!("shaders/parallel_dot.wgsl"),
        };
        buffers.push(&out_buf);
        let start = std::time::Instant::now();
        let sub_in = self.con.compute_gpu::<T>(shader, &mut buffers, dis, 1)?;
        self.con.poll_execution(sub_in);
        println!("Elapsed time shader computation: {:?}", start.elapsed());
        self.con.get_data::<T>(&out_buf)
    }

    pub fn compute_cosine_simularity<T>(
        &self,
        images: &Vec<Vec<T>>,
        re: &Vec<Vec<T>>,
        abs: &Vec<Vec<T>>,
        dis: (u32, u32, u32),
        shader: &str,
    ) -> Result<Vec<T>, WgpuContextError>
    where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        let size = (images.len() * re.len() * std::mem::size_of::<T>()) as wgpu::BufferAddress;
        let buffers = [images, re, abs]
            .iter()
            .map(|i| Self::flatten_content(i))
            .map(|i| self.con.storage_buf(&i).expect("Failed to create buffer"))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        let out_buf = self.con.read_write_buf(size)?;
        buffers.push(&out_buf);

        let _ = self.con.compute_gpu::<T>(shader, &mut buffers, dis, 1)?;
        let res = self.con.get_data::<T>(&out_buf).unwrap();

        let mut buffers = vec![&out_buf];
        let max_out_buf = self.con.read_write_buf(size / 500)?;
        buffers.push(&max_out_buf);
        let _ = self.con.compute_gpu::<T>(
            include_str!("shaders/parallel_max_pool.wgsl"),
            &mut buffers,
            dis,
            1,
        )?;
        let res = self.con.get_data::<T>(&max_out_buf).unwrap();
        Ok(res)
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
