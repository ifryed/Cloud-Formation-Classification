import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt


def prepareData(img_folder: str = "data/mini_data", img_size: int = 32, sample_size=3000):
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

    np.random.shuffle(training_data)

    X = []
    y = []

    factor = 8
    img_downsize = img_size // factor
    for o_img, label in training_data:
        folded_img = np.zeros((img_downsize, img_downsize, factor ** 2, 1))

        for i in range(factor ** 2):
            y_ind = i // factor
            x_ind = i - y_ind * factor
            folded_img[:, :, i, 0] = o_img[y_ind::factor, x_ind::factor, 0]

        X.append(folded_img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2)
