import os
import hickle


# def clean_directory_of_empty_pairs(directory):
#     for id in range(40, 67):
#         print(f"Starting for id {id}")

#         input_files = [f.path for f in os.scandir(directory) if f"inputs{id}_" in f.name]
#         output_files = [f.replace("inputs", "outputs") for f in input_files]

#         for input_file, output_file in zip(input_files, output_files):
#             input_data = hickle.load(input_file)
#             output_data = hickle.load(output_file)

#             if len(input_data) == 0 or len(output_data) == 0:
#                 print(f"Removing empty pair: {input_file}/{output_file}")
#                 print(f'Found shapes: {input_data.shape}, {output_data.shape}')
#                 os.remove(input_file)
#                 os.remove(output_file)


# directory = '/home/kage/smashbot_workspace/dataset/hickle_shuffle'
# clean_directory_of_empty_pairs(directory)


# import os
# import random
# import hickle
# import numpy as np

# def group_and_save_files_flat(directory, output_directory, group_size=10):
#     """
#     Groups hickle files that start with 'inputs{seq_len}_' into groups of size N, 
#     concatenates their numpy arrays, and saves them directly into the output directory.
    
#     :param directory: The directory where the original files are located.
#     :param output_directory: The directory where the grouped files will be saved.
#     :param group_size: The number of files in each group.
#     """
#     os.makedirs(output_directory, exist_ok=True)
    
#     # Iterate over a range of possible sequence lengths
#     for seq_len in range(3, 4):
#         input_files = [f.path for f in os.scandir(directory) if f"inputs{seq_len}_" in f.name]
#         output_files = [f.replace("inputs", "outputs") for f in input_files]
        
#         # Shuffle the lists together
#         combined_files = list(zip(input_files, output_files))
#         random.shuffle(combined_files)
#         input_files, output_files = zip(*combined_files)
        
#         # Process each group
#         for i in range(0, len(input_files), group_size):
#             print(f"Processing group {i//group_size} for sequence length {seq_len}")
#             grouped_files_in = input_files[i:i+group_size]
#             grouped_files_out = output_files[i:i+group_size]
            
#             # Collect input and output data before concatenation
#             input_data_list = [hickle.load(file) for file in grouped_files_in]
#             output_data_list = [hickle.load(file) for file in grouped_files_out]
            
#             # Concatenate all at once
#             concatenated_inputs = np.concatenate(input_data_list, axis=0)
#             concatenated_outputs = np.concatenate(output_data_list, axis=0)
            
#             # Save concatenated files
#             combined_input_filename = f"inputs{seq_len}_{i//group_size}.hkl"
#             combined_output_filename = f"outputs{seq_len}_{i//group_size}.hkl"
            
#             hickle.dump(concatenated_inputs, os.path.join(output_directory, combined_input_filename), mode='w', compression='gzip')
#             hickle.dump(concatenated_outputs, os.path.join(output_directory, combined_output_filename), mode='w', compression='gzip')

# # Example of how to use the function
# directory = "/home/kage/smashbot_workspace/dataset/hickle_action_shuffle"
# output_directory = "/home/kage/smashbot_workspace/dataset/hickle_action_shuffle/hickle_combined"
# group_size = 3  # Specify how many files you want in each group

# group_and_save_files_flat(directory, output_directory, group_size)
import os
import random
import hickle
import numpy as np
from multiprocessing import Pool

def process_group(group_data):
    """
    Worker function to process each group of files.
    """
    print(f"Processing group {group_data[-1]}")
    grouped_files_in, grouped_files_out, output_directory, seq_len, group_id = group_data
    
    # Collect input and output data before concatenation
    input_data_list = [hickle.load(file) for file in grouped_files_in]
    output_data_list = [hickle.load(file) for file in grouped_files_out]
    
    # Concatenate all at once
    concatenated_inputs = np.concatenate(input_data_list, axis=0)
    concatenated_outputs = np.concatenate(output_data_list, axis=0)
    
    # Save concatenated files
    combined_input_filename = f"inputs{seq_len}_{group_id}.hkl"
    combined_output_filename = f"outputs{seq_len}_{group_id}.hkl"
    
    hickle.dump(concatenated_inputs, os.path.join(output_directory, combined_input_filename), mode='w', compression='gzip')
    hickle.dump(concatenated_outputs, os.path.join(output_directory, combined_output_filename), mode='w', compression='gzip')

def group_and_save_files_flat(directory, output_directory, group_size=10):
    """
    Groups hickle files that start with 'inputs{seq_len}_' into groups of size N, 
    concatenates their numpy arrays, and saves them directly into the output directory using multiprocessing.
    """
    os.makedirs(output_directory, exist_ok=True)
    # Set up multiprocessing pool
    pool = Pool(processes=4)
    
    # Iterate over a range of possible sequence lengths
    for seq_len in range(3, 4):
        input_files = [f.path for f in os.scandir(directory) if f"inputs{seq_len}_" in f.name]
        output_files = [f.replace("inputs", "outputs") for f in input_files]
        
        # Shuffle the lists together
        combined_files = list(zip(input_files, output_files))
        random.shuffle(combined_files)
        input_files, output_files = zip(*combined_files)
        
        # Prepare data for each group
        tasks = [
            (input_files[i:i+group_size], output_files[i:i+group_size], output_directory, seq_len, i//group_size)
            for i in range(0, len(input_files), group_size)
        ]
        
        # Process each group in parallel
        pool.map(process_group, tasks)
    
    # Close the pool and wait for work to finish
    pool.close()
    pool.join()

# Example of how to use the function
directory = "/home/kage/smashbot_workspace/dataset/hickle_action_shuffle"
output_directory = "/home/kage/smashbot_workspace/dataset/hickle_action_shuffle/hickle_combined"
group_size = 3  # Specify how many files you want in each group

group_and_save_files_flat(directory, output_directory, group_size)