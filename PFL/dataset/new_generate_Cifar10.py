#Class 0 increases monotonically at a fixed proportion.
import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms

from utils.dataset_utils import check, separate_data, save_file


# Fix random seeds for reproducibility.
random.seed(1)
np.random.seed(1)

num_clients = 20
dir_path = "Cifar10/"


# Fixed class-0 sample counts for each client in the training set.
CLASS0_TRAIN_COUNTS = [
    1580, 0, 20, 40, 60,
    80, 100, 120, 140, 160,
    180, 200, 220, 240, 260,
    280, 300, 320, 340, 360
]


# Fixed class-0 sample counts for each client in the test set.
CLASS0_TEST_COUNTS = [
    316, 0, 4, 8, 12,
    16, 20, 24, 28, 32,
    36, 40, 44, 48, 52,
    56, 60, 64, 68, 72
]


def fixed_allocate_one_class(images, labels, target_class, counts, num_clients, split_name):
    """
    Assign a specified fixed count of the target_class to every client.
    """
    counts = [int(x) for x in counts]

    # Check whether the count list matches the number of clients.
    if len(counts) != num_clients:
        raise ValueError(
            f"The length of counts for {split_name} does not equal num_clients: "
            f"{len(counts)} != {num_clients}"
        )

    labels = np.asarray(labels)
    class_idx = np.where(labels == target_class)[0]
    np.random.shuffle(class_idx)

    need_total = sum(counts)
    real_total = len(class_idx)

    # The predefined counts must exactly match the available samples.
    if need_total != real_total:
        raise ValueError(
            f"Mismatch in allocated quantity for class {target_class} within {split_name}: "
            f"Sum of counts = {need_total}, Actual sample count = {real_total}"
        )

    client_X = []
    client_y = []

    start = 0
    for client_id in range(num_clients):
        cnt = counts[client_id]
        chosen_idx = class_idx[start:start + cnt]

        client_X.append(images[chosen_idx])
        client_y.append(labels[chosen_idx])

        start += cnt

    return client_X, client_y


def remove_target_class_from_clients(X, y, target_class):
    # Remove the target class from each client's original partition.
    new_X = []
    new_y = []

    for client_id in range(len(y)):
        x_i = np.asarray(X[client_id])
        y_i = np.asarray(y[client_id])

        keep_mask = y_i != target_class

        new_X.append(x_i[keep_mask])
        new_y.append(y_i[keep_mask])

    return new_X, new_y


def split_rest_data_inside_each_client(X, y, train_ratio=0.75, shuffle=True):
    # Split each client's remaining classes into train and test sets.
    train_X = []
    train_y = []
    test_X = []
    test_y = []

    for client_id in range(len(y)):
        x_i = np.asarray(X[client_id])
        y_i = np.asarray(y[client_id])

        if len(y_i) == 0:
            train_X.append(x_i)
            train_y.append(y_i)
            test_X.append(x_i)
            test_y.append(y_i)
            continue

        indices = np.arange(len(y_i))

        if shuffle:
            np.random.shuffle(indices)

        x_i = x_i[indices]
        y_i = y_i[indices]

        train_len = int(len(y_i) * train_ratio)

        train_X.append(x_i[:train_len])
        train_y.append(y_i[:train_len])

        test_X.append(x_i[train_len:])
        test_y.append(y_i[train_len:])

    return train_X, train_y, test_X, test_y


def concat_and_shuffle(x_parts, y_parts):
    # Concatenate multiple data parts and shuffle the result.
    real_x_parts = []
    real_y_parts = []

    for x, y in zip(x_parts, y_parts):
        y = np.asarray(y)
        if len(y) > 0:
            real_x_parts.append(np.asarray(x))
            real_y_parts.append(y)

    if len(real_x_parts) == 0:
        raise ValueError("Concatenated data is empty, please check the data allocation.")

    x_all = np.concatenate(real_x_parts, axis=0)
    y_all = np.concatenate(real_y_parts, axis=0)

    perm = np.random.permutation(len(y_all))
    x_all = x_all[perm]
    y_all = y_all[perm]

    return x_all, y_all


def build_statistic(client_train_y, client_test_y, num_classes):
    # Build client-level label statistics for save_file().
    statistic = []

    for client_id in range(len(client_train_y)):
        y_i = np.concatenate(
            [np.asarray(client_train_y[client_id]), np.asarray(client_test_y[client_id])],
            axis=0
        )

        stat_i = []
        for cls in range(num_classes):
            cnt = int(np.sum(y_i == cls))
            if cnt > 0:
                stat_i.append((int(cls), cnt))

        statistic.append(stat_i)

    return statistic


def print_client_distribution(train_y, test_y, num_classes):
    # Print the final label distribution of each client.

    total_train = 0
    total_test = 0

    global_train_counts = [0 for _ in range(num_classes)]
    global_test_counts = [0 for _ in range(num_classes)]

    for client_id in range(len(train_y)):
        y_tr = np.asarray(train_y[client_id])
        y_te = np.asarray(test_y[client_id])

        total_train += len(y_tr)
        total_test += len(y_te)

        train_counts = []
        test_counts = []

        for cls in range(num_classes):
            tr_cnt = int(np.sum(y_tr == cls))
            te_cnt = int(np.sum(y_te == cls))

            global_train_counts[cls] += tr_cnt
            global_test_counts[cls] += te_cnt

            if tr_cnt > 0:
                train_counts.append((cls, tr_cnt))
            if te_cnt > 0:
                test_counts.append((cls, te_cnt))

        print(f"Client {client_id}")
        print(f"  train_size : {len(y_tr)}")
        print(f"  test_size  : {len(y_te)}")
        print(f"  train labels: {train_counts}")
        print(f"  test  labels: {test_counts}")
        print(
            f"  class 0 train: actual={int(np.sum(y_tr == 0))}, "
            f"expected={CLASS0_TRAIN_COUNTS[client_id]}"
        )
        print(
            f"  class 0 test : actual={int(np.sum(y_te == 0))}, "
            f"expected={CLASS0_TEST_COUNTS[client_id]}"
        )
        print("-" * 100)

    print("total_train:", total_train)
    print("total_test :", total_test)
    print("total_all  :", total_train + total_test)

    print("global train counts:", global_train_counts)
    print("global test  counts:", global_test_counts)

    print("Total train Class 0 count:", global_train_counts[0])
    print("Total test Class 0 count:", global_test_counts[0])


def split_class0_fixed_keep_original_rest_distribution(
    train_images,
    train_labels,
    test_images,
    test_labels,
    num_clients,
    num_classes,
    niid,
    balance,
    partition,
    class_per_client=2,
    train_ratio=0.75,
    verbose=True
):
    # Generate a dataset where class 0 follows fixed counts,
    # while the remaining classes keep the original non-IID distribution.

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    # Combine the original CIFAR-10 train and test sets.
    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Partition all data using the original PFLlib strategy.
    all_X, all_y, _ = separate_data(
        (all_images, all_labels),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=class_per_client
    )

    # Remove original class-0 samples from each client.
    rest_X, rest_y = remove_target_class_from_clients(
        all_X,
        all_y,
        target_class=0
    )

    # Split the remaining classes into train and test sets.
    rest_train_X, rest_train_y, rest_test_X, rest_test_y = split_rest_data_inside_each_client(
        rest_X,
        rest_y,
        train_ratio=train_ratio,
        shuffle=True
    )

    # Allocate class-0 training samples using predefined counts.
    class0_train_X, class0_train_y = fixed_allocate_one_class(
        images=train_images,
        labels=train_labels,
        target_class=0,
        counts=CLASS0_TRAIN_COUNTS,
        num_clients=num_clients,
        split_name="train"
    )

    # Allocate class-0 test samples using predefined counts.
    class0_test_X, class0_test_y = fixed_allocate_one_class(
        images=test_images,
        labels=test_labels,
        target_class=0,
        counts=CLASS0_TEST_COUNTS,
        num_clients=num_clients,
        split_name="test"
    )

    final_train_X = []
    final_train_y = []
    final_test_X = []
    final_test_y = []

    # Combine fixed class-0 data with the remaining classes for each client.
    for client_id in range(num_clients):
        x_tr, y_tr = concat_and_shuffle(
            [class0_train_X[client_id], rest_train_X[client_id]],
            [class0_train_y[client_id], rest_train_y[client_id]]
        )

        x_te, y_te = concat_and_shuffle(
            [class0_test_X[client_id], rest_test_X[client_id]],
            [class0_test_y[client_id], rest_test_y[client_id]]
        )

        final_train_X.append(x_tr)
        final_train_y.append(y_tr)

        final_test_X.append(x_te)
        final_test_y.append(y_te)

    train_data = []
    test_data = []

    # Convert data into the format required by save_file().
    for client_id in range(num_clients):
        train_data.append({
            "x": final_train_X[client_id],
            "y": final_train_y[client_id]
        })

        test_data.append({
            "x": final_test_X[client_id],
            "y": final_test_y[client_id]
        })

    statistic = build_statistic(final_train_y, final_test_y, num_classes)

    if verbose:
        print_client_distribution(final_train_y, final_test_y, num_classes)

    return train_data, test_data, statistic


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    # Create the dataset directory if it does not exist.
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    # Skip generation if the dataset already exists.
    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the original CIFAR-10 training set.
    trainset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata",
        train=True,
        download=True,
        transform=transform
    )

    # Download and load the original CIFAR-10 test set.
    testset = torchvision.datasets.CIFAR10(
        root=dir_path + "rawdata",
        train=False,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=len(trainset.data),
        shuffle=False
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=len(testset.data),
        shuffle=False
    )

    # Load the whole training set into memory.
    for _, train_data_batch in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data_batch

    # Load the whole test set into memory.
    for _, test_data_batch in enumerate(testloader, 0):
        testset.data, testset.targets = test_data_batch

    train_images = trainset.data.cpu().detach().numpy()
    train_labels = trainset.targets.cpu().detach().numpy()

    test_images = testset.data.cpu().detach().numpy()
    test_labels = testset.targets.cpu().detach().numpy()

    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    num_classes = len(set(all_labels))

    print(f"Number of classes: {num_classes}")

    # Generate the final federated dataset with fixed class-0 allocation.
    train_data, test_data, statistic = split_class0_fixed_keep_original_rest_distribution(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        num_clients=num_clients,
        num_classes=num_classes,
        niid=niid,
        balance=balance,
        partition=partition,
        class_per_client=2,
        train_ratio=0.75,
        verbose=True
    )

    # Save the generated dataset files.
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
    # Read command-line arguments.
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)