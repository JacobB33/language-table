from ml_collections import ConfigDict
from language_table.environments import blocks
def get_config():
    config = ConfigDict()
    config.num_evals_per_reward = 4
    config.save_dir = "/home/jacob/projects/semantic_world_modeling/language-table/language_table/test_ds_long_horizon"
    config.random  = False
    config.max_episode_steps = 250
    config.seed_offset = 2000
    config.target_height = 180
    config.target_width = 320
    config.block_mode = blocks.LanguageTableBlockVariants.BLOCK_8
    config.save_video = False
    config.debug = False
    return config
