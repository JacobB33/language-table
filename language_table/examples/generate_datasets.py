from httpx import get
from language_table.examples.config_generate import get_config
from language_table.examples.data_gen_multi_process import generate_data

from ml_collections import ConfigDict
from language_table.environments import blocks


config = get_config()
combos = [
    # ("red_moon", "green_star"),
    ("yellow_star", "blue_cube"),
    ("yellow_pentagon", "red_moon"),
    ("green_cube", "blue_moon"),
    ("red_pentagon", "blue_moon")
    
]
for block_combo in combos:
    config = ConfigDict()
    config.num_evals_per_reward = 800
    config.save_dir = "/home/jacob/projects/semantic_world_modeling/language-table/language_table/datasets/train_diff_policy_2"
    config.random  = False
    config.max_episode_steps = 180
    config.seed_offset = 0
    config.target_height = 180
    config.target_width = 320
    config.block_mode = blocks.LanguageTableBlockVariants.BLOCK_8
    config.save_video = True
    config.debug = False
    config.block_combo = block_combo
    config.save_dir = f"/home/jacob/projects/semantic_world_modeling/language-table/language_table/datasets/expert_diffusion/{block_combo[0]}_{block_combo[1]}"
    generate_data(config.save_dir, config)