from pathlib import Path

from ml_collections import ConfigDict
repo_root = Path(__file__).parent.parent
exp_name = "train_balanced_fixed"
def get_config():
    config = ConfigDict()
    config.pickle_dir = f"{repo_root}/datasets/{exp_name}/demos"
    config.save_path = f"{repo_root}/datasets/{exp_name}/{exp_name}_new.hdf5"
    config.max_horizon = 60
    config.alpha = 1.6
    config.beta = 1.0
    config.num_horizon_samples = 12
    return config
