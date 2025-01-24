from ml_collections import ConfigDict
from language_table.environments import blocks
def get_config():
    config = ConfigDict()
    config.num_evals_per_reward = 3
    config.save_dir = "/home/jacob/projects/semantic_world_modeling/language-table/language_table/outputs_novel_scene"
    config.random  = False
    config.max_episode_steps = 200
    config.target_height = 180
    config.target_width = 320
    config.block_mode = blocks.LanguageTableBlockVariants.NOVEL_8
    return config
