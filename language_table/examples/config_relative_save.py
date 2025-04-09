from pathlib import Path

from ml_collections import ConfigDict
repo_root = Path(__file__).parent.parent
exp_name = "test_ds_long_horizon"
def get_config():
    config = ConfigDict()
    config.pickle_dir = f"{repo_root}/{exp_name}/demos"
    config.save_path = f"{repo_root}/{exp_name}/dataset.hdf5"
    config.max_horizon = 80
    config.alpha = 1.6
    config.beta = 1.0
    config.num_horizon_samples = 5
    return config
