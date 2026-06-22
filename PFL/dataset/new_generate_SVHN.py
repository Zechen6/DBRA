#Class 0 increases monotonically at a fixed proportion.(SVHN)
import numpy as np
import os
import sys
import random
import scipy.io as sio

try:
    from dataset.utils.dataset_utils import check, separate_data, save_file
except ImportError:
    from utils.dataset_utils import check, separate_data, save_file


random.seed(1)
np.random.seed(1)

num_clients = 20
dir_path = "SVHN/"


# Fixed class-0 counts for the SVHN training set. Total = 4948.
CLASS0_TRAIN_COUNTS = [
    1564, 0, 20, 40, 59,
    79, 99, 119, 139, 158,
    178, 198, 218, 237, 257,
    277, 297, 317, 336, 356
]


CLASS0_TEST_COUNTS = [
    551, 0, 7, 14, 21,
    28, 35, 42, 49, 56,
    63, 70, 77, 84, 91,
    98, 104, 111, 118, 125
]


def fixed_allocate_one_class(images, labels, target_class, counts, num_clients, split_name):
    """
    Assign a fixed number of target-class samples to each client.
    """
    counts = [int(x) for x in counts]

    labels = np.asarray(labels)
    class_idx = np.where(labels == target_class)[0]
    np.random.shuffle(class_idx)

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
    """
    Remove the target class from each client's Dirichlet partition.
    """
    new_X = []
    new_y = []

    for client_id in range(len(y)):
        x_i = np.asarray(X[client_id])
        y_i = np.asarray(y[client_id])

        keep_mask = y_i != target_class

        new_X.append(x_i[keep_mask])
        new_y.append(y_i[keep_mask])

    return new_X, new_y


def split_rest_data_inside_each_client_exact(X, y, target_train_total, shuffle=True):
    """
    Split non-target-class data within each client while matching the global train size.
    """
    num_clients = len(y)

    sizes = np.array([len(y_i) for y_i in y], dtype=int)
    total_rest = int(sizes.sum())


    raw_train_sizes = sizes * (target_train_total / total_rest)
    train_sizes = np.floor(raw_train_sizes).astype(int)

    remain = int(target_train_total - train_sizes.sum())
    frac = raw_train_sizes - train_sizes

    order = np.argsort(-frac)

    for i in order[:remain]:
        train_sizes[i] += 1

    train_X = []
    train_y = []
    test_X = []
    test_y = []

    for client_id in range(num_clients):
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

        tr_len = int(train_sizes[client_id])

        train_X.append(x_i[:tr_len])
        train_y.append(y_i[:tr_len])

        test_X.append(x_i[tr_len:])
        test_y.append(y_i[tr_len:])

    return train_X, train_y, test_X, test_y


def concat_and_shuffle(x_parts, y_parts):
    real_x_parts = []
    real_y_parts = []

    for x, y in zip(x_parts, y_parts):
        y = np.asarray(y)
        if len(y) > 0:
            real_x_parts.append(np.asarray(x))
            real_y_parts.append(y)

    x_all = np.concatenate(real_x_parts, axis=0)
    y_all = np.concatenate(real_y_parts, axis=0)

    perm = np.random.permutation(len(y_all))
    x_all = x_all[perm]
    y_all = y_all[perm]

    return x_all, y_all


def build_statistic(client_train_y, client_test_y, num_classes):
    statistic = []

    for client_id in range(len(client_train_y)):
        y_i = np.concatenate(
            [
                np.asarray(client_train_y[client_id]),
                np.asarray(client_test_y[client_id])
            ],
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
    """
    Print the final label distribution of each client.
    """
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

    print("=" * 100)


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
    class_per_client=4,
    verbose=True
):
    """
    Fix class-0 allocation and keep the remaining classes under the original non-IID split.
    """

    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)

    all_images = np.concatenate([train_images, test_images], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    all_X, all_y, _ = separate_data(
        (all_images, all_labels),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=class_per_client
    )

    rest_X, rest_y = remove_target_class_from_clients(
        all_X,
        all_y,
        target_class=0
    )

    target_train_total_rest = len(train_labels) - sum(CLASS0_TRAIN_COUNTS)

    rest_train_X, rest_train_y, rest_test_X, rest_test_y = split_rest_data_inside_each_client_exact(
        rest_X,
        rest_y,
        target_train_total=target_train_total_rest,
        shuffle=True
    )

    class0_train_X, class0_train_y = fixed_allocate_one_class(
        images=train_images,
        labels=train_labels,
        target_class=0,
        counts=CLASS0_TRAIN_COUNTS,
        num_clients=num_clients,
        split_name="train"
    )

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


def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    f = "/home/zyq/PFLlib/dataset"

    train = sio.loadmat(f + "/train_32x32.mat")
    test = sio.loadmat(f + "/test_32x32.mat")

    train_images = train["X"]
    test_images = test["X"]

    train_labels = train["y"].flatten()
    test_labels = test["y"].flatten()

    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)

    train_labels[train_labels == 10] = 0
    test_labels[test_labels == 10] = 0

    train_images = np.transpose(train_images, (3, 2, 0, 1))
    test_images = np.transpose(test_images, (3, 2, 0, 1))

    all_labels = np.concatenate([train_labels, test_labels], axis=0)
    num_classes = len(set(all_labels))

    print(f"Number of classes: {num_classes}")
    print(f"SVHN train total: {len(train_labels)}")
    print(f"SVHN test  total: {len(test_labels)}")
    print(f"SVHN all   total: {len(all_labels)}")

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
        class_per_client=4,
        verbose=True
    )

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