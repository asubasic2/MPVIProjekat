import cv2
from skimage import io
import numpy as np
from skimage import transform
import json
from PIL import Image, ImageDraw
import glob
import os
import ntpath
from skimage import exposure


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def croppimg(path, to=""):
    for filename in glob.glob(path + '/*.png'):
        end = path_leaf(filename)
        im = cv2.imread(filename)
        im = cv2.resize(im, (256, 256))
        # im = transform.resize(im, (128, 128))
        # print("/"+end)
        cv2.imwrite(to + "/" + end, im)


def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized


def imgmod(path, to=""):
    for filename in glob.glob(path + '/*.png'):
        end = path_leaf(filename)
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(im, 100, 200)
        # im = constrastLimit(im)
        cv2.imwrite(to + "/" + end, edges)


def load_dataset(test, ann_test, train, ann_tran):
    croppimg(test, "Dataset/Znakovi_cropped_test")
    croppimg(ann_test, "Dataset/Znakovi_seg_cropped_test")
    croppimg(train, "Dataset/Znakovi_cropped_train")
    croppimg(ann_tran, "Dataset/Znakovi_seg_cropped_train")


load_dataset("Dataset/Znakovi_test", "Dataset/Znakovi_segmentation_test", "Dataset/Znakovi_train",
             "Dataset/Znakovi_segmentation_train")


def annotation(path, path_to):
    for filename in glob.glob(path + '/*.png'):
        end = path_leaf(filename)
        end_json = end[:len(end) - 4]
        print(end_json)
        cv2.waitKey(0)
        im = cv2.imread(filename)
        label = -1
        with open('Dataset/Annotations/' + end_json + ".json") as json_file:
            data = json.load(json_file)
            for p in data['shapes']:
                label = int(p['label'])

        for i in range(len(im)):
            for j in range(len(im[i])):
                if str(im[i][j]) != "[0 0 0]":
                    im[i][j] = [label, label, label]

        cv2.imwrite(path_to + "/" + end, im)


def class0(path, path_to):
    for filename in glob.glob(path + '/*.png'):
        end = path_leaf(filename)
        end_json = end[:len(end) - 4]
        print(end_json)
        cv2.waitKey(0)
        im = cv2.imread(filename)
        label = -1
        for i in range(len(im)):
            for j in range(len(im[i])):
                im[i][j] = [0, 0, 0]

        cv2.imwrite(path_to + "/" + end, im)


# class0("Dataset","Dataset/Znakovi_segmentation_train")
annotation("Dataset/Znakovi_seg_cropped_test", "Dataset/Ann_test")
annotation("Dataset/Znakovi_seg_cropped_train", "Dataset/Ann_train")
# imgmod("Dataset/Znakovi_cropped_train", "Dataset/Modified")
