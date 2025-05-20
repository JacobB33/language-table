import os
import pickle
from copy import copy
from typing import List

import h5py
import numpy as np
from tqdm.auto import tqdm

from language_table.environments.blocks import LanguageTableBlockVariants
from language_table.environments.lang_table_data_generation import LanguageTableDataGeneration
from absl import app
from ml_collections import config_flags
from language_table.examples.relative_questions import (generate_did_block_move_questions,
                                                        generate_relative_block_block_questions,
                                                        generate_relative_peg_block_questions,
                                                        generate_block_move_direction_questions,
                                                        generate_peg_move_questions
                                                        )

_CONFIG = config_flags.DEFINE_config_file(
    "config", "/home/jacob/projects/semantic_world_modeling/language-table/language_table/examples/config_relative_save.py", "Training configuration.", lock_config=True)

def get_files(trajectory_folder):
    files = [file for file in os.listdir(trajectory_folder) if file.endswith(".pkl")]
    files.sort()
    return files


def process_one_file(config, file_path):
    data = pickle.load(open(file_path, "rb"))
    # env = LanguageTableDataGeneration(
    #     block_mode=LanguageTableBlockVariants.BLOCK_8,
    #     reward_factory=None,
    #     seed=0,  # Ensure different seeds per worker
    #     delay_reward_steps=5
    #
    # )
    qa_horizons = []
    for i in range(len(data['observations'])-10):
        # sample horizons using a beta distribution that
        # horizons = (np.random.beta(config.alpha, config.beta, config.num_horizon_samples)
        #             *  min(config.max_horizon, len(data['observations']) - i - 1))
        horizons = np.random.uniform(0, min(len(data['observations']) - i - 1, config.max_horizon), config.num_horizon_samples)

        horizons = set(round(h) for h in horizons)
        hdict = {}
        for horizon in horizons:
            previous_poses = data['block_states'][i]
            final_poses = data['block_states'][i+horizon]
            questions: List = copy(data['qa_pairs'][i+horizon])
            weights: List = copy(data['weights'][i+horizon])
            if horizon > 0:
                to_add = [
                    generate_relative_block_block_questions(past_states=previous_poses,current_states=final_poses, num_questions=8),
                    generate_relative_peg_block_questions(past_states=previous_poses, current_states=final_poses),
                    generate_block_move_direction_questions(past_states=previous_poses, current_states=final_poses, num_questions=8),
                    generate_did_block_move_questions(past_states=previous_poses, current_states=final_poses),
                    generate_peg_move_questions(past_states=previous_poses, current_states=final_poses)
                ]
                questions.extend(
                    [x for question_list, _ in to_add for x in question_list]
                )
                weights.extend(
                    [x for _, weights in to_add for x in weights]
                )
            w_sum = sum(weights)
            hdict[horizon] = {
                'start_idx': i,
                'end_idx': i + horizon,
                'questions': questions,
                'weights': [w / w_sum for w in weights],
            }
            assert len(hdict[horizon]['questions']) == len(hdict[horizon]['weights']), f"{i}: {len(hdict[horizon]['questions'])} {len(hdict[horizon]['weights'])}"
        qa_horizons.append(hdict)
    frames = data['frames']
    actions = data['actions']
    return frames, actions, qa_horizons



def save_data(config, processed_data_list):
    save_idx = 0
    with h5py.File(config.save_path, 'w') as f:
        for processed_data in processed_data_list:
            frames, actions, qa_horizon_dicts = processed_data
            if len(qa_horizon_dicts) == 0:
                print("skipping trajectory since not enough data to satisfy min horizon")
                continue

            data_grp = f.create_group(f"traj_{save_idx}")
            save_idx += 1

            data_grp.create_dataset("frames", data=frames)
            data_grp.create_dataset("actions", data=actions)
            qa_horizon_grp = data_grp.create_group(f"horizon_start")
            for h_start, horizon_dict in enumerate(qa_horizon_dicts):
                start_grp = qa_horizon_grp.create_group(f"{h_start}")
                for h_len, qa_horizon_dict in horizon_dict.items():
                    specific_grp = start_grp.create_group(f"horizon_len_{h_len}")
                    qs = [x[0] for x in qa_horizon_dict["questions"]]
                    answers = [x[1] for x in qa_horizon_dict["questions"]]
                    types = [x[2] for x in qa_horizon_dict["questions"]]
                    specific_grp.create_dataset("questions", data=qs)
                    specific_grp.create_dataset("types", data=types)
                    specific_grp.create_dataset("weights", data=qa_horizon_dict['weights'])
                    specific_grp.create_dataset("answers", data=answers)
                    specific_grp.create_dataset("start_idx", data=qa_horizon_dict['start_idx'])
                    specific_grp.create_dataset("end_idx", data=qa_horizon_dict['end_idx'])




def main(argv):
    config = _CONFIG.value
    files = [f for f in os.listdir(config.pickle_dir) if f.endswith(".pkl")]
    processed_data = []
    for file in tqdm(files):
        processed_data.append(process_one_file(config, os.path.join(config.pickle_dir, file)))
    save_data(config, processed_data)


if __name__ == "__main__":
    app.run(main)
