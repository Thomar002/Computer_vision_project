"""Configuration for the dehazing project."""
import os

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")

# RESIDE outdoor (SOTS test set)
RESIDE_SOTS_HAZY = os.path.join(DATASETS_DIR, "RESIDE-dataset", "outdoor", "hazy")
RESIDE_SOTS_GT = os.path.join(DATASETS_DIR, "RESIDE-dataset", "outdoor", "clear")

# RESIDE OUT (for training)
RESIDE_OUT_TRAIN_GT = os.path.join(DATASETS_DIR, "RESIDE OUT", "RESIDE OUT", "train", "GT")
RESIDE_OUT_TEST_HAZY = os.path.join(DATASETS_DIR, "RESIDE OUT", "RESIDE OUT", "test", "hazy")
RESIDE_OUT_TEST_GT = os.path.join(DATASETS_DIR, "RESIDE OUT", "RESIDE OUT", "test", "GT")

# O-HAZE
OHAZE_HAZY = os.path.join(DATASETS_DIR, "O-HAZY", "hazy")
OHAZE_GT = os.path.join(DATASETS_DIR, "O-HAZY", "GT")

# Output paths
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Training hyperparameters
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "image_size": 256,
    "num_workers": 4,
}

# DCP parameters
DCP_CONFIG = {
    "patch_size": 15,
    "omega": 0.95,
    "t0": 0.1,
    "guided_filter_radius": 60,
    "guided_filter_eps": 1e-3,
}
