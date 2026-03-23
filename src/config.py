from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "preprocessed_dataset"
KAGGLE_DIR = PROJECT_ROOT / "kaggle-dataset"
IMAGES_DIR = PROJECT_ROOT / "images"

TRAIN_PATH = DATA_DIR / "train_dataset.csv"
VALIDATION_PATH = DATA_DIR / "validation_dataset.csv"
TEST_PATH = DATA_DIR / "test_dataset.csv"

# Constants
RANDOM_STATE = 42
TRAIN_SPLIT_RATIO = 0.7
ZERO_SALES_KEEP_FRACTION = 0.2

# LightGBM hyperparameters
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}
LIGHTGBM_NUM_ROUNDS = 100
