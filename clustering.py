import argparse
import os

import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt

from utils import prepareData


class NOT_SKLEARN_KNN(object):
    def __init__(self, n_neightbors=5):
        self.k_neigh = n_neightbors

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.cats = list(set(self.labels))

    def predict(self, new_data):
        lbls = []
        for samp in new_data:
            nn_idxs = np.power(samp - self.data, 2).sum(axis=1).argsort()[:self.k_neigh]

            lbls_lst = self.labels[nn_idxs]
            lbls_count = np.array([sum(lbls_lst == x) for x in self.cats])
            best_lbl = self.cats[lbls_count.argmax()]
            lbls.append(best_lbl)
        return lbls


def getKNN(nn_model, images: np.ndarray, labels: np.ndarray):
    imgs_vecs = nn_model.predict(images)
    knn = NOT_SKLEARN_KNN(n_neightbors=5)
    knn.fit(imgs_vecs, labels)
    return knn


def main(model_path: str, img_fld: str):
    # Training the KNN
    model = keras.models.load_model(model_path)
    img_h = img_w = model.inputs[0].shape[1]
    train_x, test_x, train_y, test_y = prepareData(
        img_fld,
        img_h,
        sample_size=-30,
        normalize=True)
    epoch = len(train_x)

    print("Building KNN model..")
    knn = getKNN(model, train_x, train_y)

    print("Predicting the Test dataset..")
    test_vecs = model.predict(test_x)
    test_pred = knn.predict(test_vecs)
    accuracy = np.asarray(test_pred == test_y).sum() / len(test_y)
    print("Accuracy: %f" % accuracy)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    parser = argparse.ArgumentParser(description='Train KNN')
    parser.add_argument('--model', dest="model", type=str, required=True,
                        help='The trained model to load')
    parser.add_argument('--images', dest="img_folder", type=str, required=True,
                        help='Location of the images')

    args = parser.parse_args()

    main(args.model, args.img_folder)
