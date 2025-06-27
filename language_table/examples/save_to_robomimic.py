import os
import pickle

import h5py
from tqdm.auto import tqdm


def get_files(trajectory_folder):
    files = [file for file in os.listdir(trajectory_folder) if file.endswith(".pkl")]
    files.sort()
    return files


def process_one_file(file_path):
    data = pickle.load(open(file_path, "rb"))
    frames = data['frames'][:-1]
    actions = data['actions']
    return frames, actions



def save_data(save_path, processed_data_list):
    save_idx = 0
    with h5py.File(save_path, 'w') as f:
        main_grp = f.create_group("data")
        for processed_data in processed_data_list:
            frames, actions = processed_data

            data_grp = main_grp.create_group(f"demo_{save_idx}")
            save_idx += 1
            
            data_grp.create_dataset("actions", data=actions)
            obs_grp = data_grp.create_group("obs")
            obs_grp.create_dataset("frames", data=frames)


def main(dataset_dir, save_path):    
    files = [f for f in os.listdir(dataset_dir) if f.endswith(".pkl")]
    processed_data = []
    for file in tqdm(files):
        processed_data.append(process_one_file(os.path.join(dataset_dir, file)))
    save_data(save_path, processed_data)


if __name__ == "__main__":
    dataset_path = "/home/jacob/projects/semantic_world_modeling/language-table/language_table/datasets/train_balanced_fixed_wtype/demos"
    save_path = "/home/jacob/projects/semantic_world_modeling/language-table/language_table/datasets/train_balanced_fixed_wtype/train_balanced_fixed_wtype_robomimic.hdf5"
    main(dataset_path, save_path)
