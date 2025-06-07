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

block_names = [
    'blue_moon',
    # 'blue_cube',
    # 'green_cube',
    'green_star',
    'yellow_star',
    'yellow_pentagon'
    'red_moon',
    # 'red_pentagon',
]

locations = [
    # 'center',
    # 'center_left',
    # 'center_right',
    'bottom',
    'bottom_left',
    'bottom_right',
    # 'top_right',
    # 'top_left',
    # 'top',
]

def generate_block_to_block():
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

def generate_block_to_location():
    config = ConfigDict()
    config.num_evals_per_reward = 200
    config.random  = False
    config.max_episode_steps = 180
    config.seed_offset = 0
    config.target_height = 180
    config.target_width = 320
    config.block_mode = blocks.LanguageTableBlockVariants.BLOCK_8
    config.save_video = True
    config.debug = False
    config.block = None
    config.location = None
    config.block_combo = None
    config.save_dir = f"/gscratch/weirdlab/yanda/semantic_world_modeling/datasets/expert_diffusion3_200/"
    generate_data("empty", config, block_names, locations)

# generate_block_to_block()
generate_block_to_location()