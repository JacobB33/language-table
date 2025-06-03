from ml_collections import ConfigDict
from language_table.environments import blocks
def get_config():
    config = ConfigDict()
    config.num_evals_per_reward = 800
    config.save_dir = "/gscratch/weirdlab/yanda/swm/language_table/datasets/train_diff_policy_3"
    config.random  = False
    config.max_episode_steps = 180
    config.seed_offset = 0
    config.target_height = 180
    config.target_width = 320
    config.block_mode = blocks.LanguageTableBlockVariants.BLOCK_8
    config.save_video = True
    config.debug = False
    config.block_combo = None
    return config
