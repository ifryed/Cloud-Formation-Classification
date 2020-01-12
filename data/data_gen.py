import pandas as pd
import numpy as np
import os
import cv2

from utils import rle_to_mask


def genDataWithMasks(data):
    output = os.path.join('data', 'mini_data')

    for t in ["Fish", "Flower", "Sugar", "Gravel"]:
        lbl_folder = os.path.join(output, t)
        os.makedirs(lbl_folder, exist_ok=True)
        os.makedirs(os.path.join(lbl_folder, 'masks'), exist_ok=True)

    for row in data.iterrows():
        img_name, img_type = row[1][0].split('_')
        img_path = os.path.join(images_path, img_name)

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        pix = rle_to_mask(row[1][1], w, h)

        img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        pix = cv2.resize(pix, (0, 0), fx=0.25, fy=0.25)

        cv2.imwrite(os.path.join(output, img_type, img_name), img)
        cv2.imwrite(os.path.join(output, img_type, 'masks', img_name), pix)


def getBB(pix_mask):
    pix_mask = cv2.morphologyEx(pix_mask, cv2.MORPH_OPEN, np.ones((10, 10)))
    contours, hierarchy = cv2.findContours(pix_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbs = []
    for c in contours:
        blob = c.squeeze()
        blob_xy = np.array([blob.min(axis=0),
                            blob.max(axis=0)])

        if np.any(blob_xy[1, :] - blob_xy[0, :] < 256):
            continue
        bbs.append(blob_xy)
    return bbs


def genDataBB(data):
    # Saving the images at size 350X525
    # Each image contains only one type
    output = os.path.join('mini_data')
    os.makedirs(output, exist_ok=True)

    out_w, out_h = 256, 256
    name_counter = 0

    for t in ["Fish", "Flower", "Sugar", "Gravel"]:
        lbl_folder = os.path.join(output, t)
        os.makedirs(lbl_folder, exist_ok=True)

    for row in data.iterrows():
        img_name, img_type = row[1][0].split('_')
        img_path = os.path.join(images_path, img_name)

        img = cv2.imread(img_path)
        h, w, _ = img.shape
        mask = rle_to_mask(row[1][1], w, h)

        bbs = getBB(mask)
        for bb in bbs:
            crop = img[bb[0, 1]:bb[1, 1],
                   bb[0, 0]:bb[1, 0]]
            crop = cv2.resize(crop, (out_h, out_w))

            img_name = "%05d.png" % name_counter
            cv2.imwrite(os.path.join(output, img_type, img_name), crop)
            name_counter += 1


def main():
    data = pd.read_csv('train.csv')
    data = data[data['EncodedPixels'].isnull() == False]
    print("Classes:", data.keys())

    genDataBB(data)

    print("Done!")


if __name__ == "__main__":
    images_path = "train_images"
    main()
