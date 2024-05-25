import os
import pickle

def load_and_verify_pickle(directory):
    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    print(f"Found {len(pickle_files)} pickle files in directory {directory}.")

    for file in pickle_files:
        filepath = os.path.join(directory, file)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                print(f"Loaded {file} successfully with {len(data)} records.")
                
                # Print sample data from the first record to verify structure
                if data:
                    sample = data[1]
                    print("Sample data from the first record:")
                    print(sample)
                else:
                    print("No data in this file.")
        except Exception as e:
            print(f"Failed to load {file} due to an error: {e}")

# Usage example:
directory = '/home/kage/smashbot_workspace/dataset/pickle_files'  # Adjust to your directory path
load_and_verify_pickle(directory)