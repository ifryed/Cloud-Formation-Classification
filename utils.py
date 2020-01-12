import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def NOT_SK_LEARN_train_test_split(
        X: np.ndarray,
        Y: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 24, shuffle=True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Splits the data to train/test subsets, can shuffle the data too.
    :param X: The data
    :param Y: The labels
    :param test_size: The test size in percentage
    :param random_state: random seed
    :param shuffle: True to shuffle the data before the split
    :return: train_x, test_x, train_y, test_y
    """
    if shuffle:
        np.random.seed(random_state)
    data_size = len(X)
    idxs = np.array([x for x in range(data_size)])
    np.random.shuffle(idxs)

    X = X[idxs]
    Y = Y[idxs]

    test_size_idx = int(data_size * test_size)

    train_x = X[:-test_size_idx]
    train_y = Y[:-test_size_idx]

    test_x = X[test_size_idx:]
    test_y = Y[test_size_idx:]

    return train_x, test_x, train_y, test_y


def rle_to_mask(rle_string: str, width: int, height: int, norm=False) -> np.ndarray:
    """
    convert RLE(run length encoding) string to numpy array

    Parameters:
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns:
    numpy.array: numpy array of the mask
    """

    rows, cols = height, width
    lbl_val = 255.
    if norm:
        lbl_val = 1.

    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        img = np.zeros(rows * cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index + length] = lbl_val
        img = img.reshape(cols, rows)
        img = img.T
        return img


def prepareData(img_folder: str = "data/mini_data", img_size: int = 32, sample_size=3000, normalize=False):
    CATEGORIES = os.listdir(img_folder)
    training_data = []
    max_data_sampeles = min([len(x) for x in [os.listdir(os.path.join(img_folder, y)) for y in CATEGORIES]])
    sample_size = min(sample_size, max_data_sampeles)
    sample_size = sample_size if sample_size > 0 else max_data_sampeles
    for category in CATEGORIES:
        path = os.path.join(img_folder, category)
        class_num = CATEGORIES.index(category)
        c = sample_size
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size, img_size))
                img_h, img_w = img_array.shape
                img_array = img_array.reshape((img_h, img_w, 1)).astype(np.float32)
                training_data.append([img_array, class_num])
                c -= 1
                if c == 0:
                    break
            except IOError as e:  # in the interest in keeping the output clean...
                pass

    X = []
    y = []

    factor = 4
    img_downsize = img_size // factor
    for o_img, label in training_data:
        # folded_img = np.zeros((img_downsize, img_downsize, factor ** 2, 1))
        #
        # for i in range(factor ** 2):
        #     y_ind = i // factor
        #     x_ind = i - y_ind * factor
        #     folded_img[:, :, i, 0] = o_img[y_ind::factor, x_ind::factor, 0]

        X.append(o_img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    if normalize:
        # mu = X.mean(0)
        # img_std = X.std(0)
        # X = (X - mu) / img_std
        X = X / 255.0

    return NOT_SK_LEARN_train_test_split(X, y, test_size=0.3, random_state=24)


def prepareSegData(img_list_file: str = "data/train.csv", img_folder: str = "data/mini_data", img_size: int = 32,
                   sample_size=3000, normalize=False):
    kCATAGORIES = {'Fish': 0, 'Gravel': 1, 'Flower': 2, 'Sugar': 3}

    data = pd.read_csv(img_list_file)
    data = data[data['EncodedPixels'].isnull() == False]
    image_label_data = dict()

    samp_counter = sample_size * len(kCATAGORIES)
    for row in tqdm(data.iterrows()):
        img_name, img_type = row[1][0].split('_')
        img_path = os.path.join(img_folder, img_name)

        img = cv2.imread(img_path).astype(np.float)
        h, w, _ = img.shape
        mask = rle_to_mask(row[1][1], w, h, norm=normalize)

        img = cv2.resize(img, (img_size, img_size))
        if normalize:
            img = img / 255.
        mask = cv2.resize(mask, (img_size, img_size))
        h, w, _ = img.shape

        if img_name not in image_label_data.keys():
            multi_mask = np.zeros((h, w, len(kCATAGORIES)), dtype=np.float)
            image_label_data[img_name] = [img, multi_mask]
        image_label_data[img_name][1][:, :, kCATAGORIES[img_type]] = mask

        samp_counter -= 1
        if samp_counter == 0:
            break

    image_label_data = list(image_label_data.values())
    np.random.shuffle(image_label_data)

    X = []
    y = []
    for img, seg_label in image_label_data:
        X.append(
            img.reshape(img_size, img_size, 3)
        )
        y.append(
            seg_label.reshape(img_size, img_size, 4)
        )

    X = np.array(X).astype(np.float)
    y = np.array(y).astype(np.float)

    return NOT_SK_LEARN_train_test_split
