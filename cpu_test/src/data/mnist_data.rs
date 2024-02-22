use mnist::*;
use ndarray::{s, Array, Array3};

pub struct MnistImages {
    pub train_images: Vec<Vec<Vec<f32>>>,
}

pub fn load_mnist_dataset(num_images: u32) -> MnistImages {
    let training_set_length = 50_000u32;
    let Mnist { trn_img, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(training_set_length)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let num_images = num_images.min(training_set_length);

    let train_data = Array3::from_shape_vec((50_000, 28, 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f32 / 255.0);

    let images: Vec<Vec<Vec<f32>>> = (0..num_images)
        .map(|image_num| {
            let image = train_data.slice(s![image_num as usize, .., ..]);
            // Create a new 29x29 array filled with 0s for padding.
            let mut padded_image = Array::zeros((29, 29));
            // Copy the original image data into the top-left corner of the padded image.
            padded_image.slice_mut(s![0..28, 0..28]).assign(&image);
            // Convert the padded image to Vec<Vec<f32>>
            padded_image.outer_iter().map(|row| row.to_vec()).collect()
        })
        .collect();

    MnistImages {
        train_images: images,
    }
}
