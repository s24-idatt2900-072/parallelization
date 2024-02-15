use crate::wgpu_context::WgpuContext;

/// Extractor performs feature extraction using WGPU operations.
pub struct Extractor {
    con: WgpuContext,
}

impl Extractor {
    /// Creates a new instance of the Extractor with a WGPU context.
    pub fn new() -> Self {
        let con = WgpuContext::new().expect("Failed to create context");
        Self { con }
    }

    /// Initiates the feature extraction process using provided data and parameters.
    ///
    /// # Arguments
    ///
    /// * `a` - Input data matrix A.
    /// * `b` - Input data matrix B.
    /// * `out` - Output matrix for storing computed features.
    /// * `chunk` - Size of the computation chunk.
    /// * `filter_chunk` - Size of the filter chunk for maxpooling.
    pub fn feature_extraction<T>(
        a: &Vec<Vec<T>>,
        b: &Vec<Vec<T>>,
        out: &mut Vec<Vec<T>>,
        chunk: usize,
        filter_chunk: usize,
    ) where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        Extractor::new().get_features(a, b, out, chunk, filter_chunk);
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
        out: &mut Vec<Vec<T>>,
        chunk: usize,
        filter_chunk: usize,
    ) where
        T: bytemuck::Pod,
        T: std::fmt::Debug,
    {
        let inner_size = a.get(0).expect("No input provided").len();
        if b.iter().all(|i| i.len() != inner_size) {
            panic!("Can't compute with different lengths");
        }
        // Memory size for the output data
        let size = (std::mem::size_of::<T>() * ((b.len() * a.len() * inner_size) / chunk))
            as wgpu::BufferAddress;
        // Instantiates buffers for computating.
        let buffers = [a, b]
            .iter()
            .map(|i| Self::flatten_content(i))
            .map(|i| self.con.read_only_buf(&i))
            .collect::<Vec<wgpu::Buffer>>();
        let mut buffers = buffers.iter().map(|b| b).collect::<Vec<&wgpu::Buffer>>();
        let info_buf = self.con.read_only_buf::<u32>(&vec![
            inner_size as u32,
            b.len() as u32,
            a.len() as u32,
            chunk as u32,
            filter_chunk as u32,
        ]);
        buffers.insert(0, &info_buf);
        let out_buf = self.con.read_write_buf(size);
        buffers.push(&out_buf);

        let dis_size = Extractor::get_dispatch_size(
            a.len() as i32,
            b.len() as i32,
            inner_size as i32,
            chunk as i32,
        );
        self.con.compute_gpu::<T>(
            include_str!("shaders/feature_extraction.wgsl"),
            &mut buffers,
            dis_size,
            3,
        );
        self.con.get_data(out, &out_buf);
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
        T: std::fmt::Debug,
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
        (x as u32, y as u32, 1)
    }
}
