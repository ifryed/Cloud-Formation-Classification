import os

import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.neighbors import KNeighborsClassifier

from utils import prepareData


def getKNN(nn_model, images: np.ndarray, labels: np.ndarray):
    knn = KNeighborsClassifier(n_neighbors=5)#, weights='distance')
    imgs_vecs = nn_model.predict(images)
    knn.fit(imgs_vecs, labels)
    return knn


def main(model_path: str, img_fld: str):
    # Training the KNN
    model = keras.models.load_model(model_path)
    img_h = img_w = 32
    train_x, test_x, train_y, test_y = prepareData(img_fld, img_h, sample_size=-30)
    epoch = len(train_x)

    knn = getKNN(model, train_x, train_y)

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
