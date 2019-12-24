import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def prepareData(img_folder: str = "data/mini_data", img_size: int = 32, sample_size=3000):
    CATEGORIES = os.listdir(img_folder)
    training_data = []
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
                training_data.append([img_array, class_num])  # add this to our training_data
                c -= 1
                if c == 0:
                    break
            except IOError as e:  # in the interest in keeping the output clean...
                pass

    np.random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2)
