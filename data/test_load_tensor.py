import os
import torch



def load_and_verify_pickle(directory):
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    print(f"Found {len(pickle_files)} pickle files in directory {directory}.")

    for file in pickle_files:
        filepath = os.path.join(directory, file)
        data = torch.load(filepath)
        print(data[0][0], data[0][1])
        print(f"Lenght of data in {file}: {len(data)}")
       

# Usage example:
directory = '/home/kage/smashbot_workspace/dataset/pt_files'  # Adjust to your directory path
load_and_verify_pickle(directory)