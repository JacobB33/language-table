import h5py

def combine_hdf5_files(file1_path, file2_path, output_path):
    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2, h5py.File(output_path, 'w') as fout:
        # Copy all groups from file1
        for key in f1.keys():
            f1.copy(key, fout)

        # Determine the next trajectory index to avoid collisions
        max_idx = max(int(k.split('_')[1]) for k in fout.keys() if k.startswith("traj_"))

        # Copy all groups from file2 with new names
        for key in f2.keys():
            if key.startswith("traj_"):
                orig_idx = int(key.split('_')[1])
                new_key = f"traj_{max_idx + 1}"
                f2.copy(key, fout, name=new_key)
                max_idx += 1
        print(f"Combined data from {file1_path} and {file2_path} into {output_path} with {max_idx + 1} trajectories.")

if __name__ == "__main__":
    file1 = "/home/jacob/projects/semantic_world_modeling/ogbench/datasets/train_swm_noise_dataset/train_swm_noise_dataset.hdf5"
    file2 = "/home/jacob/projects/semantic_world_modeling/ogbench/datasets/train_swm_play_dataset/train_swm_play_dataset.hdf5"
    output_file = "/home/jacob/projects/semantic_world_modeling/ogbench/datasets/train_swm_play_dataset/og_final_mixed.hdf5"
    combine_hdf5_files(file1, file2, output_file)