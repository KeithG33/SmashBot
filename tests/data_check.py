# import os
# import hickle
# import numpy as np

# DATA_DIR = '/home/kage/smashbot_workspace/dataset/SlippiGames/hickle'
# NUM_FILES_TO_PROCESS = 5  # Number of files to process
# NUM_RANDOM_FRAMES = 3     # Number of random frames to show from each file

# # Iterate over the first NUM_FILES_TO_PROCESS files in the directory
# for i, data_file in enumerate(os.scandir(DATA_DIR)):
#     if i >= NUM_FILES_TO_PROCESS:
#         break
#     data = hickle.load(data_file.path)
#     print(f"Data file: {data_file.name}, Data shape: {data.shape}")
    
#     # Show the initial and final frame of the data
#     print(f"Initial frame: {data[0]}")
#     print(f"Final frame: {data[-1]}")
    
#     # Select NUM_RANDOM_FRAMES random indices (excluding the first and last to avoid repetition)
#     random_indices = np.random.choice(range(1, data.shape[0] - 1), NUM_RANDOM_FRAMES, replace=False)
#     for index in random_indices:
#         print(f"Random frame {index}: {data[index]}")
#     print()  # Add an empty line for better readability between files




import os
import random 
import hickle
DATA_DIR = '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/outputs3_batch23_part0.hkl'

data = hickle.load(DATA_DIR)

size = len(data)
print(data.shape)

indx = random.randint(0, size)

print(f"Random frame {indx} digital: {data[indx,:5]}")
print(f"Random frame {indx} analog: {data[indx,5:]}")
