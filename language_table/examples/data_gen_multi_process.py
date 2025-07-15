import multiprocessing
import os
import pickle
import random
from functools import partial
import copy
from tqdm import tqdm

import mediapy as mediapy_lib
import numpy as np
import tensorflow as tf
from absl import app
from absl import logging
from ml_collections import config_flags
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

from language_table.environments import language_table
from language_table.environments import lang_table_data_generation
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation, point2block
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers
import os

n_procs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) 
print(f"Using {n_procs} processes")

_CONFIG = config_flags.DEFINE_config_file(
    "config", "/gscratch/weirdlab/yanda/swm/language_table/examples/config_generate.py", "Training configuration.", lock_config=True)
# _WORKDIR = flags.DEFINE_string("workdir", _CONFIG.value.save_dir, "Evaluation result directory.")

tf.config.experimental.set_visible_devices([], "GPU")
def generate_episode(reward_name, reward_factory, config, ep_num, max_episode_steps, workdir):
    """Generates a single episode data."""
    env = lang_table_data_generation.LanguageTableDataGeneration(
        block_mode=config.block_mode,
        reward_factory=reward_factory,
        seed=ep_num + config.seed_offset,  # Ensure different seeds per worker
        delay_reward_steps = 5,
        block_combo=config.block_combo,
        block=config.block,
        location=config.location
    )
    env = gym_wrapper.GymWrapper(env)
    env = env_wrappers.CentralCropImageWrapper(
        env,
        target_width=config.target_width,
        target_height=config.target_height,
        random_crop_factor=1
    )
    env = tfa_wrappers.HistoryWrapper(
        env, history_length=1, tile_first_step_obs=True
    )

    oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
        env, use_ee_planner=True
    )
    plan_success = False

    while not plan_success:
        ts = env.reset()
        raw_state = env.compute_state()
        plan_success = oracle_policy.get_plan(raw_state)
        if not plan_success:
            logging.info(
                "Resetting environment because the "
                "initialization was invalid (could not find motion plan)."
            )
    metadata = {
        "reward_name": reward_name,
        "instruction_str": env._instruction_str,
        "start_block": env._start_block,
        "oracle_target_block": env._oracle_target_block,
        "oracle_target_translation": env._oracle_target_translation,
        "target_absolute_location": env._target_absolute_location,
        "target_relative_location": env._target_relative_location,
    }
    frames = [(ts.observation["rgb"][0] * 255).astype(np.uint8)]

    initial_state = env.get_pybullet_state()
    actions = []
    observations = [ts]
    block_states = [env.get_block_states()]
    qa_pairs = []
    weights = []
    to_add =  [env.get_block_touching_questions(),
                    env.get_relative_block2block_questions(number_of_questions=8),
                    env.get_peg_block_questions(), # there are 8 of them
                    env.get_block_to_board_questions(number_of_questions=8)]
    
    qa_pairs.append([q for q_set, _ in to_add for q in q_set])
    # Flatten and collect all weights
    weights.append([w for _, w_set in to_add for w in w_set])

        
    episode_steps = 0
    while not ts.is_last():
        raw_state = env.compute_state()
        if config.random:
            policy_step = np.random.uniform(-.025, .025, size=2)
            # if np.random.rand() < 0.9:
            #     policy_step = oracle_policy._get_action_for_block_target(raw_state)
        else:
            policy_step = oracle_policy._get_action_for_block_target(raw_state)

        ts = env.step(policy_step)
        frames.append((ts.observation["rgb"][0] * 255).astype(np.uint8))
        actions.append(policy_step)
        observations.append(ts)
        to_add =  [env.get_block_touching_questions(),
                        env.get_relative_block2block_questions(number_of_questions=8),
                        env.get_peg_block_questions(), # there are 8 of them
                        env.get_block_to_board_questions(number_of_questions=8)]
        
        qa_pairs.append([q for q_set, _ in to_add for q in q_set])
        # Flatten and collect all weights
        weights.append([w for _, w_set in to_add for w in w_set])
        block_states.append(env.get_block_states())

        episode_steps += 1
        if episode_steps > max_episode_steps:
            break

    success_str = "success" if env.succeeded else "failure"
    if env.succeeded:
        logging.info("Episode %d: success.", ep_num)
    else:
        logging.info("Episode %d: failure.", ep_num)
        if not config.random:
            # this episode failed so we want to generate a new one. If we are in random mode, we want to keep the episode.
            return False
    if config.save_video and ep_num == 0:
        # Write out video of rollout.
        video_path = os.path.join(workdir, "videos/", f"{reward_name}_{ep_num}_{success_str}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        mediapy_lib.write_video(video_path, frames, fps=10)

    # Save the trajectory
    trajectory = {
        "metadata": metadata,
        "initial_state": initial_state,
        "actions": actions,
        "observations": observations,
        "frames": frames,
        "qa_pairs": qa_pairs,
        "weights": weights,
        "block_states": block_states
    }
    trajectory_path = os.path.join(workdir, "demos/", f"{reward_name}_{ep_num}_{success_str}.pkl")
    if not os.path.exists(os.path.dirname(trajectory_path)):
        os.makedirs(os.path.dirname(trajectory_path), exist_ok=True)
    pickle.dump(trajectory, open(trajectory_path, "wb"))
    return True


def get_state_for_relative_questions(env: language_table.LanguageTable):
    state = env.compute_state()
    return state


def generate_episode_wrapper(reward_name, reward_factory, config, ep_num, max_episode_steps, workdir):
    while not generate_episode(reward_name, reward_factory, config, ep_num, max_episode_steps, workdir):
        pass
    return True

def generate_data(workdir, config, blocks=None, locations=None):
    """Evaluates the given checkpoint and writes results to workdir."""
    video_dir = os.path.join(workdir, "videos")
    if not tf.io.gfile.exists(video_dir):
        tf.io.gfile.makedirs(video_dir)

    rewards = {
        "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
    }

    num_evals_per_reward = config.num_evals_per_reward
    max_episode_steps = config.max_episode_steps

    # If blocks and locations are not provided, use single default config
    if blocks is None:
        blocks = [config.block]
    if locations is None:
        locations = [config.location]

    total_configs = len(rewards) * len(blocks) * len(locations) * num_evals_per_reward
    
    if config.debug:
        pbar = tqdm(total=total_configs, desc="Generating episodes (debug mode)")
        for reward_name, reward_factory in rewards.items():
            for block in blocks:
                for location in locations:
                    for i in range(num_evals_per_reward):
                        # Create a deep copy of config with current block and location
                        run_config = copy.deepcopy(config)
                        run_config.block = block
                        run_config.location = location
                        if workdir == "empty":
                            directory = os.path.join(run_config.save_dir, f"{block}_to_{location}")
                            run_config.save_dir = directory

                        generate_episode_wrapper(reward_name, reward_factory, run_config, i, max_episode_steps, directory)
                        pbar.update(1)
        pbar.close()
    else:
        with multiprocessing.Pool(processes=n_procs) as pool:
            for reward_name, reward_factory in rewards.items():
                # Create all run configurations
                run_configs = []
                for block in blocks:
                    for location in locations:
                        for i in range(num_evals_per_reward):
                            run_config = copy.deepcopy(config)
                            run_config.block = block
                            run_config.location = location
                            directory = workdir
                            if workdir == "empty":
                                directory = os.path.join(run_config.save_dir, f"{block}_to_{location}")
                                run_config.save_dir = directory

                            # Create a tuple of arguments for each run
                            run_configs.append((reward_name, reward_factory, run_config, i, max_episode_steps, directory))
                
                with tqdm(total=len(run_configs), desc=f"{reward_name}", position=0, leave=True) as pbar:
                    results = [
                        pool.apply_async(generate_episode_wrapper, args=args)
                        for args in run_configs
                    ]
                    for res in results:
                        res.get()  # blocks and raises errors
                        pbar.update(1)
                
                print(f"Finished reward: {reward_name}")
                # logging.error("Finished reward: %s", reward_name)


def main(argv):
    config = _CONFIG.value
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    generate_data(
        workdir=config.save_dir,
        config=config,
    )

if __name__ == "__main__":
    app.run(main)
