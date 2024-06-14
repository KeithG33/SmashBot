import os
import random

import hickle
import numpy as np


# def shuffle_files(input_dir, output_dir, batch_size, id='_n3_'):
#     input_files = [f.path for f in os.scandir(input_dir) if "inputs_" in f.name and id in f.name]
#     # use replace to get output files
#     output_files = [f.replace("inputs_", "outputs_") for f in input_files]

#     # Load batch_size random files at a time from input and output files and shuffle the data
#     # and save as batch_size new files in output_idr
#     while len(input_files) > 0:
#         # Load batch
#         input_batch = input_files[:batch_size]
#         output_batch = output_files[:batch_size]
#         len_input_batch = len(input_batch)

#         input_data = [hickle.load(f) for f in input_batch]
#         output_data = [hickle.load(f) for f in output_batch]

#         combined_inputs = np.concatenate(input_data, axis=0)
#         combined_outputs = np.concatenate(output_data, axis=0)

#         # Shuffle data together
#         combined_indices = np.arange(combined_inputs.shape[0])
#         np.random.shuffle(combined_indices)
#         combined_inputs = combined_inputs[combined_indices]
#         combined_outputs = combined_outputs[combined_indices]

#         # Split data back into len_input_batch files
#         input_data = np.array_split(combined_inputs, len_input_batch)
#         output_data = np.array_split(combined_outputs, len_input_batch)

#         # Save shuffled data using old filenames
#         for idx, (input, output) in enumerate(zip(input_data, output_data)):
#             input_filename = os.path.basename(input_batch[idx])
#             input_path = os.path.join(output_dir, input_filename)
#             output_path = input_path.replace("inputs_", "outputs_")
#             hickle.dump(input, input_path, mode='w', compression='gzip')
#             hickle.dump(output, output_path, mode='w', compression='gzip')

#         # Remove batch from list
#         input_files = input_files[batch_size:]
#         output_files = output_files[batch_size:]

import os
import hickle
import numpy as np
from multiprocessing import Pool


def process_files(data, batch_index):
    input_batch, output_batch, output_dir, num_save_files, id = data
    input_data = [hickle.load(f) for f in input_batch]
    output_data = [hickle.load(f) for f in output_batch]

    # Combine and shuffle data
    combined_inputs = np.concatenate(input_data, axis=0)
    combined_outputs = np.concatenate(output_data, axis=0)
    
    # assert outputs have shape (B, 10)
    assert combined_outputs.shape[1] == 10, (
        f"combined_outputs.shape[1] is {combined_outputs.shape[1]} - part of {print(output_batch)}"
    )

    assert combined_inputs.shape[-1] == 21, (
        f"combined_inputs.shape[-1] is {combined_inputs.shape[-1]} - part of {print(input_batch)}"
    ) 

    combined_indices = np.arange(combined_inputs.shape[0])
    np.random.shuffle(combined_indices)
    combined_inputs = combined_inputs[combined_indices]
    combined_outputs = combined_outputs[combined_indices]

    # Split data back into specified number of files
    num_save_files = 3 if len(input_batch) <= 3 else min(num_save_files, len(input_batch))
    input_data = np.array_split(combined_inputs, num_save_files)
    output_data = np.array_split(combined_outputs, num_save_files)

    # Save shuffled data using modified filenames
    for idx, (inp, out) in enumerate(zip(input_data, output_data)):
        input_filename = f"inputs{id}_batch{batch_index}_part{idx}.hkl"
        input_path = os.path.join(output_dir, input_filename)
        output_path = input_path.replace("inputs", "outputs")
        hickle.dump(inp, input_path, mode='w', compression='gzip')
        hickle.dump(out, output_path, mode='w', compression='gzip')

def shuffle_files(input_dir, output_dir, batch_size, num_processes, num_save_files, id=3):
    input_files = [f.path for f in os.scandir(input_dir) if f"inputs{id}_" in f.name]
    output_files = [f.replace("inputs", "outputs") for f in input_files]

    if len(input_files) == 0:
        print(f"No files found in {input_dir} with id {id}")
        return
    
    # Shuffle input and output files together by index
    combined_files = list(zip(input_files, output_files))
    random.shuffle(combined_files)
    input_files, output_files = zip(*combined_files)

    # Create batches of files for each process
    file_batches = [(input_files[i:i+batch_size], output_files[i:i+batch_size], output_dir, num_save_files, id) 
                    for i in range(0, len(input_files), batch_size)]
    
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_files, [(batch, i) for i, batch in enumerate(file_batches)])


if __name__ == '__main__':
    for id in range(3, 67):
        shuffle_files(
            '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle_combined',
            '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle_shuffle',
            10,
            num_processes=3,
            num_save_files=8,
            id=id
        )

        
