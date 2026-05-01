"""Configuration for the dehazing project."""
import os

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
DATASETS_DIR = os.path.join(BASE_DIR, "Datasets")

# ITS-RESIDE (Indoor Training Set)
ITS_TRAIN_HAZY = os.path.join(DATASETS_DIR, "ITS-reside", "hazy")
ITS_TRAIN_GT = os.path.join(DATASETS_DIR, "ITS-reside", "clear")

# OTS-RESIDE (Outdoor Training Set)
OTS_TRAIN_HAZY = os.path.join(DATASETS_DIR, "OTS-reside", "hazy")
OTS_TRAIN_GT = os.path.join(DATASETS_DIR, "OTS-reside", "clear")

# SOTS (Testing)
SOTS_INDOOR_HAZY = os.path.join(DATASETS_DIR, "SOTS", "indoor", "hazy")
SOTS_INDOOR_GT = os.path.join(DATASETS_DIR, "SOTS", "indoor", "clear")
SOTS_OUTDOOR_HAZY = os.path.join(DATASETS_DIR, "SOTS", "outdoor", "hazy")
SOTS_OUTDOOR_GT = os.path.join(DATASETS_DIR, "SOTS", "outdoor", "clear")

# O-HAZE
OHAZE_HAZY = os.path.join(DATASETS_DIR, "O-HAZY", "hazy")
OHAZE_GT = os.path.join(DATASETS_DIR, "O-HAZY", "clear")

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
