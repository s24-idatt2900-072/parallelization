import numpy as np
import logging

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

def load_and_stack_data(data_dict, count):
    """ Convert a dictionary of 29x29 arrays into a 3D stack of arrays, repeating or slicing as needed to meet the specified count. """
    all_data = np.stack(list(data_dict.values()))
    n = all_data.shape[0]
    if count <= n:
        return all_data[:count]
    full_repeats = count // n
    remaining = count % n
    return np.concatenate([all_data] * full_repeats + [all_data[:remaining]])