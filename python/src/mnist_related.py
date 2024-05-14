import numpy as np
import logging
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Load data from a CSV file. 
    File format:
    #0
    then 29 rows and 29 columns of floating numbers 
    and so on
    each 29x29 should be stored in numpy array
    """
    data_blocks = {}
    try:
        with open(filepath, 'r') as file:
            block_id = None
            current_data = []
            for line in file:
                line = line.strip()
                if line.startswith("#"):
                    if current_data:
                        if len(current_data) == 29 and all(len(row) == 29 for row in current_data):
                            data_blocks[block_id] = np.array(current_data)
                        else:
                            logging.warning(f"Data block {block_id} has incorrect dimensions and was not added.")
                        current_data = []
                    try:
                        block_id = int(line[1:])
                    except ValueError:
                        logging.error(f"Invalid block ID: {line}")
                        block_id = None
                        continue
                elif block_id is not None:
                    try:
                        data_row = [float(num) for num in line.split(',')]
                        current_data.append(data_row)
                    except ValueError as e:
                        logging.error(f"Error parsing float values in line: {line} - {e}")
            if current_data and block_id is not None and len(current_data) == 29 and all(len(row) == 29 for row in current_data):
                data_blocks[block_id] = np.array(current_data)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")

    return data_blocks

def pad_images(images, pad_value=0):
    """
    Pad images to achieve a size of 29x29 by adding 1 pixel padding to the right and bottom.

    Args:
        images (np.array): Original array of images (N, 28, 28).
        pad_value (int): Value to fill in padding.

    Returns:
        np.array: Padded images array.
    """
    # Pad images: (number of images, vertical pad, horizontal pad)
    # Pad 0 pixels to the top and left, 1 pixel to the bottom and right
    padded_images = np.pad(images, ((0, 0), (0, 1), (0, 1)), mode='constant', constant_values=pad_value)
    return padded_images


def load_and_pad_mnist_torch():
    # Define a transform to pad and convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load the dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Access the padded images (and labels, if needed)
    train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=1000, shuffle=True)
    images, labels = next(iter(train_loader))

    # Convert tensors to numpy arrays for compatibility
    images = images.numpy()
    images = np.squeeze(images)  # Remove the channel dimension for grayscale

    # Pad images to 29x29
    images_padded = pad_images(images)

    return images_padded



def save_images_to_file(images, filename):
    """
    Save padded images to a file with each image prefixed by its index as #id.

    Args:
        images (np.array): Array of images to save.
        filename (str): Output filename.
    """
    with open(filename, 'w') as f:
        for index, img in enumerate(images):
            f.write(f'#{index}\n')
            for row in img:
                f.write(','.join(format(x, '.2f') for x in row) + '\n')



def display_image_from_file(filename, image_id):
    """
    Display an image by its ID from the file.

    Args:
        filename (str): Filename where images are stored.
        image_id (int): ID of the image to display.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    start = lines.index(f'#{image_id}\n') + 1
    end = start + 29
    image = np.array([list(map(float, line.split(','))) for line in lines[start:end]])
    plt.imshow(image, cmap='gray')
    plt.show()



file_path = 'files/filters/filters_abs.csv'
data_blocks  = load_data(file_path)

padded_images = load_and_pad_mnist_torch()

save_images_to_file(padded_images, 'files/mnist/mnist_padded_29x29.csv')

display_image_from_file('files/mnist/mnist_padded_29x29.csv', 0)

