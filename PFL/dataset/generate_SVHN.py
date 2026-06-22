#Generate SVHN dataset
import numpy as np
import os
import sys
import random
import scipy.io as sio

from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)

num_clients = 20
dir_path = "SVHN/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # =========================
    # Load SVHN data
    # =========================
    f = "/home/zyq/PFLlib/dataset"

    train = sio.loadmat(f + "/train_32x32.mat")
    test = sio.loadmat(f + "/test_32x32.mat")

    train_data = train["X"]  
    test_data = test["X"]     

    train_labels = train["y"].flatten()
    test_labels = test["y"].flatten()

    train_data = np.transpose(train_data, (3, 2, 0, 1))
    test_data = np.transpose(test_data, (3, 2, 0, 1))

    dataset_image = np.concatenate([train_data, test_data], axis=0)
    dataset_label = np.concatenate([train_labels, test_labels], axis=0)

    dataset_label[dataset_label == 10] = 0

    num_classes = len(set(dataset_label))
    print(f"Number of classes: {num_classes}")

    X, y, statistic = separate_data(
        (dataset_image, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=4
    )

    train_data, test_data = split_data(X, y)

    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition
    )


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)