import argparse
import glob
import os
from random import shuffle
from shutil import move

from utils import get_module_logger

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"
TRAIN_VAL_TEST_PROP = [0.8, 0.1, 0.1]


def make_directories(base_dir):
    train_dir = os.path.join(base_dir, TRAIN_DIR)
    val_dir = os.path.join(base_dir, VAL_DIR)
    test_dir = os.path.join(base_dir, TEST_DIR)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    return train_dir, val_dir, test_dir


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    train_dir, val_dir, test_dir = make_directories(destination)
    tfrec_files = glob.glob(os.path.join(source, "*tfrecord"))
    shuffle(tfrec_files)
    threshold_train = round(len(tfrec_files) * TRAIN_VAL_TEST_PROP[0])
    for tfrec_file in tfrec_files[:threshold_train]:
        move(tfrec_file, train_dir)

    threshold_val = round(len(tfrec_files) * TRAIN_VAL_TEST_PROP[1])
    for tfrec_file in tfrec_files[threshold_train: threshold_train + threshold_val]:
        move(tfrec_file, val_dir)

    for tfrec_file in tfrec_files[threshold_train + threshold_val:]:
        move(tfrec_file, test_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    parser.add_argument("--source", required=True,
                        help="source data directory")
    parser.add_argument(
        "--destination", required=True, help="destination data directory"
    )
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.source, args.destination)
