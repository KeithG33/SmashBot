
import os
import random
import hickle as hkl
import numpy as np

# def combine_files(directory, id):
#     out_files = [f.path for f in os.scandir(directory) if f"outputs{id}_" in f.name]
#     in_files = [f.replace("outputs", "inputs") for f in out_files]
#     print(len(in_files), len(out_files))
#     # Separate smaller files (those with N around 1.5 million)
#     small_in_files = []
#     small_out_files = []
#     for infile, outfile in zip(in_files, out_files):
#         output_data = hkl.load(outfile)
#         print(f"Loaded {outfile} with shape {output_data.shape}")
#         # if output_data.shape[0] < 2_000_000:
#         #     print(f"Found small file: {outfile}")
#         #     small_in_files.append(infile)
#         #     small_out_files.append(outfile)

#     # # Combine pairs of small files
#     # i = 1
#     # for idx in range(0, len(small_in_files), 2):
#     #     if idx + 1 < len(small_in_files):
#     #         # Load paired input and output files
#     #         input_data_1 = hkl.load(small_in_files[idx])
#     #         output_data_1 = hkl.load(small_out_files[idx])
#     #         input_data_2 = hkl.load(small_in_files[idx + 1])
#     #         output_data_2 = hkl.load(small_out_files[idx + 1])

#     #         # Combine data
#     #         combined_inputs = np.concatenate((input_data_1, input_data_2), axis=0)
#     #         combined_outputs = np.concatenate((output_data_1, output_data_2), axis=0)

#     #         # Save combined files
#     #         combined_in_filename = os.path.join(directory, f"inputs{id}_combined{i}.hkl")
#     #         combined_out_filename = os.path.join(directory, f"outputs{id}_combined{i}.hkl")
#     #         hkl.dump(combined_inputs, combined_in_filename, compression="gzip", mode="w")
#     #         hkl.dump(combined_outputs, combined_out_filename, compression="gzip", mode="w")
#     #         i += 1

#     #         # Rename old files
#     #         print(f"Renaming {small_in_files[idx]} and {small_in_files[idx+1]}")
#     #         os.rename(small_in_files[idx], os.path.join(directory, "OLD_" + os.path.basename(small_in_files[idx])))
#     #         os.rename(small_out_files[idx], os.path.join(directory, "OLD_" + os.path.basename(small_out_files[idx])))
#     #         os.rename(small_in_files[idx + 1], os.path.join(directory, "OLD_" + os.path.basename(small_in_files[idx + 1])))
#     #         os.rename(small_out_files[idx + 1], os.path.join(directory, "OLD_" + os.path.basename(small_out_files[idx + 1])))



# if __name__ == "__main__":
#     directory = "/home/kage/smashbot_workspace/dataset/hickle_shuffle"
#     combine_files(directory, 7)




def combine_files_in_batches(directory, id):
    out_files = [f.path for f in os.scandir(directory) if f"outputs{id}_" in f.name and "OLD" not in f.name]
    in_files = [f.replace("outputs", "inputs") for f in out_files]
    
    paired_files = list(zip(in_files, out_files))
    random.shuffle(paired_files)
    in_files, out_files = zip(*paired_files)
    
    # Group files into batches of 10
    batch_size = 3
    i = 6
    for idx in range(0, len(in_files), batch_size):
        input_batch = in_files[idx:idx + batch_size]
        output_batch = out_files[idx:idx + batch_size]
        
        if len(input_batch) > 0:
            # Load and concatenate data
            combined_inputs = np.concatenate([hkl.load(f) for f in input_batch], axis=0)
            combined_outputs = np.concatenate([hkl.load(f) for f in output_batch], axis=0)

            # Save combined files
            combined_in_filename = os.path.join(directory, f"inputs{id}_combined{i}.hkl")
            combined_out_filename = os.path.join(directory, f"outputs{id}_combined{i}.hkl")
            hkl.dump(combined_inputs, combined_in_filename, compression="gzip", mode="w")
            hkl.dump(combined_outputs, combined_out_filename, compression="gzip", mode="w")
            i += 1

            # Rename old files
            for infile, outfile in zip(input_batch, output_batch):
                os.rename(infile, os.path.join(directory, "OLD_" + os.path.basename(infile)))
                os.rename(outfile, os.path.join(directory, "OLD_" + os.path.basename(outfile)))

if __name__ == "__main__":
    directory = "/home/kage/smashbot_workspace/dataset/hickle_shuffle"
    combine_files_in_batches(directory, id=3)
