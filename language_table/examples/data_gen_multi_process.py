import multiprocessing
import os
import pickle
import random
from functools import partial

import mediapy as mediapy_lib
import numpy as np
import tensorflow as tf
from absl import app
from absl import logging
from ml_collections import config_flags
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation, point2block
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers

_CONFIG = config_flags.DEFINE_config_file(
    "config", "/home/jacob/projects/semantic_world_modeling/language-table/language_table/examples/config.py", "Training configuration.", lock_config=True)
# _WORKDIR = flags.DEFINE_string("workdir", _CONFIG.value.save_dir, "Evaluation result directory.")

tf.config.experimental.set_visible_devices([], "GPU")
def generate_episode(reward_name, reward_factory, config, ep_num, max_episode_steps, workdir):
    """Generates a single episode data."""
    env = language_table.LanguageTable(
        block_mode=config.block_mode,
        reward_factory=reward_factory,
        seed=ep_num  # Ensure different seeds per worker
    )
    env = gym_wrapper.GymWrapper(env)
    # env = env_wrappers.ClipTokenWrapper(env)
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

    frames = [(ts.observation["rgb"][0] * 255).astype(np.uint8)]

    initial_state = env.get_pybullet_state()
    actions = []
    observations = [ts]
    block_states = [env.get_block_states()]
    qa_pairs = [
        random.sample(env.get_block_touching_questions(), 8) +
        env.get_relative_block2block_questions(number_of_questions=8) +
        env.get_peg_block_questions() + # there are 8 of them
        env.get_block_to_board_questions(number_of_questions=8)
    ]

    episode_steps = 0
    while not ts.is_last():
        raw_state = env.compute_state()
        if config.random:
            policy_step = np.random.uniform(-.018, .018, size=2)
            if np.random.rand() < 0.9:
                policy_step = oracle_policy._get_action_for_block_target(raw_state)
        else:
            policy_step = oracle_policy._get_action_for_block_target(raw_state)

        ts = env.step(policy_step)
        frames.append((ts.observation["rgb"][0] * 255).astype(np.uint8))
        actions.append(policy_step)
        observations.append(ts)
        qa_pairs.append(
            random.sample(env.get_block_touching_questions(), 8) +
            env.get_relative_block2block_questions(number_of_questions=8) +
            env.get_peg_block_questions() + # there are 8 of them
            env.get_block_to_board_questions(number_of_questions=8)
        )
        block_states.append(env.get_block_states())

        episode_steps += 1
        if episode_steps > max_episode_steps:
            break

    success_str = "success" if env.succeeded else "failure"
    if env.succeeded:
        logging.info("Episode %d: success.", ep_num)
    else:
        logging.info("Episode %d: failure.", ep_num)
        return False
    if config.save_video:
        # Write out video of rollout.
        video_path = os.path.join(workdir, "videos/", f"{reward_name}_{ep_num}_{success_str}.mp4")
        mediapy_lib.write_video(video_path, frames, fps=10)

    # Save the trajectory
    trajectory = {
        "initial_state": initial_state,
        "actions": actions,
        "observations": observations,
        "frames": frames,
        "qa_pairs": qa_pairs,
        "block_states": block_states
    }
    trajectory_path = os.path.join(workdir, "demos/", f"{reward_name}_{ep_num}_{success_str}.pkl")
    if not os.path.exists(os.path.dirname(trajectory_path)):
        os.makedirs(os.path.dirname(trajectory_path))
    pickle.dump(trajectory, open(trajectory_path, "wb"))
    return True


def get_state_for_relative_questions(env: language_table.LanguageTable):
    state = env.compute_state()
    return state


def generate_episode_wrapper(reward_name, reward_factory, config, ep_num, max_episode_steps, workdir):
    while not generate_episode(reward_name, reward_factory, config, ep_num, max_episode_steps, workdir):
        pass


def generate_data(workdir, config):
    """Evaluates the given checkpoint and writes results to workdir."""
    video_dir = os.path.join(workdir, "videos")
    if not tf.io.gfile.exists(video_dir):
        tf.io.gfile.makedirs(video_dir)

    rewards = {
        "blocktoblock": block2block.BlockToBlockReward,
        "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
        "blocktoblockrelativelocation": block2block_relative_location.BlockToBlockRelativeLocationReward,
        "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
        "separate": separate_blocks.SeparateBlocksReward,
        "peg to block": point2block.PointToBlockReward,
    }

    num_evals_per_reward = config.num_evals_per_reward
    max_episode_steps = config.max_episode_steps

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        for reward_name, reward_factory in rewards.items():
            worker_fn = partial(
                generate_episode_wrapper, reward_name, reward_factory, config, max_episode_steps=max_episode_steps, workdir=workdir
            )
            pool.map(worker_fn, range(num_evals_per_reward))
            logging.error("Finished reward: %s", reward_name)


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
