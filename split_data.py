import os
from shutil import copy, rmtree
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    random.seed(0)
    val_ratio = 0.2
    test_ratio = 0.3
    data_root = os.path.join(os.getcwd(), "npy_data")
    dataset_root = os.path.join(os.getcwd(), 'fatty_liver_dataset')
    assert os.path.exists(data_root), "path '{}' does not exist.".format(data_root)

    liver_class = [cla for cla in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, cla))]

    train_root = os.path.join(dataset_root, "train")
    mk_file(train_root)
    for cla in liver_class:
        mk_file(os.path.join(train_root, cla))

    val_root = os.path.join(dataset_root, "val")
    mk_file(val_root)
    for cla in liver_class:
        mk_file(os.path.join(val_root, cla))
    test_root = os.path.join(dataset_root, "test")
    mk_file(test_root)
    for cla in liver_class:
        mk_file(os.path.join(test_root, cla))

    for cla in liver_class:
        cla_path = os.path.join(data_root, cla)
        npy = np.array(os.listdir(cla_path))
        num = len(npy)
        shuffled_indices = np.random.permutation(num)
        test_set_size = int(num * test_ratio)
        val_set_size = int(num * val_ratio)
        m = test_set_size + val_set_size
        test_indices = shuffled_indices[:test_set_size]
        val_indices = shuffled_indices[test_set_size:m]
        train_indices = shuffled_indices[m:]
        test_path = npy[test_indices]
        val_path = npy[val_indices]
        train_path = npy[train_indices]

        for index, npy in enumerate(npy):
            if npy in test_path:
                npy_path = os.path.join(cla_path, npy)
                new_path = os.path.join(test_root, cla)
                copy(npy_path, new_path)
            elif npy in val_path:
                npy_path = os.path.join(cla_path, npy)
                new_path = os.path.join(val_root, cla)
                copy(npy_path, new_path)
            else:
                npy_path = os.path.join(cla_path, npy)
                new_path = os.path.join(train_root, cla)
                copy(npy_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main()
