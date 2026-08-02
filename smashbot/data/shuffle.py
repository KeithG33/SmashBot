import os
import random
import shutil
import hickle
import numpy as np
from multiprocessing import Pool


def split_train_test(directory, output_directory):
    os.makedirs(os.path.join(output_directory, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'test'), exist_ok=True)

    for seq_len in range(3, 67):
        input_files = [f.path for f in os.scandir(directory) if f"inputs" in f.name]
        output_files = [f.replace("inputs", "outputs") for f in input_files]

        # Pair input and output files together
        paired_files = list(zip(input_files, output_files))
        random.shuffle(paired_files)  # Shuffle to randomize the order

        # Determine split ratio
        num_files = len(paired_files)

        # Check if there is only one file pair
        if num_files == 1:
            train_files = paired_files  # Move the only file pair to the train set
            test_files = []  # No files in the test set
        else:
            num_test = max(1, int(num_files * 0.15))
            num_train = num_files - num_test

            # Split files into train and test sets
            train_files = paired_files[num_test:]
            test_files = paired_files[:num_test]

        # Move files to respective directories
        for input_file, output_file in train_files:
            shutil.move(input_file, os.path.join(output_directory, 'train', os.path.basename(input_file)))
            shutil.move(output_file, os.path.join(output_directory, 'train', os.path.basename(output_file)))
        for input_file, output_file in test_files:
            shutil.move(input_file, os.path.join(output_directory, 'test', os.path.basename(input_file)))
            shutil.move(output_file, os.path.join(output_directory, 'test', os.path.basename(output_file)))


def process_files(data, batch_index):
    print(f"Processing batch {batch_index}")
    input_batch, output_batch, output_dir, num_save_files, id = data
    input_data = [hickle.load(f) for f in input_batch]
    output_data = [hickle.load(f) for f in output_batch]

    print(f"Loaded {len(input_data)} input files and {len(output_data)} output files")
    
    # Combine and shuffle data
    input_data = np.concatenate(input_data, axis=0)
    output_data = np.concatenate(output_data, axis=0)

    print(f"Combined data shapes: {input_data.shape}, {output_data.shape}")

    combined_indices = np.arange(input_data.shape[0])
    np.random.shuffle(combined_indices)
    input_data = input_data[combined_indices]
    output_data = output_data[combined_indices]

    # Split data back into specified number of files
    num_save_files = len(input_batch) if len(input_batch) <= 3 else min(num_save_files, len(input_batch))
    input_data = np.array_split(input_data, num_save_files)
    output_data = np.array_split(output_data, num_save_files)

    print(f"Split data into {len(input_data)} parts")

    # Save shuffled data using modified filenames
    for idx, (inp, out) in enumerate(zip(input_data, output_data)):
        input_filename = f"inputs_batch{batch_index}_part{idx}.hkl"
        input_path = os.path.join(output_dir, input_filename)
        output_path = input_path.replace("inputs", "outputs")
        hickle.dump(inp, input_path, mode='w', compression='gzip')
        hickle.dump(out, output_path, mode='w', compression='gzip')


def shuffle_files(input_dir, output_dir, batch_size, num_processes, num_save_files, id=3):
    input_files = [f.path for f in os.scandir(input_dir) if f"inputs" in f.name]
    output_files = [f.replace("inputs", "outputs") for f in input_files]

    if len(input_files) == 0:
        print(f"No files found in {input_dir} with id {id}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle input and output files together by index
    combined_files = list(zip(input_files, output_files))
    random.shuffle(combined_files)
    input_files, output_files = zip(*combined_files)

    # Create batches of files for each process
    file_batches = [(input_files[i:i+batch_size], output_files[i:i+batch_size], output_dir, num_save_files, id) 
                    for i in range(0, len(input_files), batch_size)]
    
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            pool.starmap(process_files, [(batch, i) for i, batch in enumerate(file_batches)])
    else:
        # Process each batch sequentially
        for batch_index, batch in enumerate(file_batches):
            process_files(batch, batch_index)



            
if __name__ == '__main__':
    # for id in range(3, 4):
    #     print(f"Starting shuffle for id {id}")
    #     shuffle_files(
    #         '/home/kage/smashbot_workspace/dataset/hickle_simple',
    #         '/home/kage/smashbot_workspace/dataset/hickle_simple_shuffle',
    #         50,
    #         num_processes=1,
    #         num_save_files=25,
    #         id=id
    #     )
    split_train_test('/home/kage/smashbot_workspace/dataset/hickle_sequence', '/home/kage/smashbot_workspace/dataset/hickle_sequence')
        
