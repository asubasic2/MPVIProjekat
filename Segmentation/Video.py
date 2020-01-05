from model.model import Cnn
from keras_segmentation.train import find_latest_checkpoint
import os
import json
from keras_segmentation.models import model_from_name
import cv2
import numpy as np


def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized


def model_from_checkpoint_path(checkpoints_path):
    assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (not latest_weights is None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](model_config['n_classes'],
                                                         input_height=model_config['input_height'],
                                                         input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


model = Cnn.build(width=256, height=256, depth=3, classes=5)
# model = model_from_checkpoint_path("path_to_checkpoints")
model.load_weights("Proba1.14")
cap = cv2.VideoCapture('Video/Stop2.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break
    if image is None:
        break
    im = np.rot90(image, k=3)
    image = np.rot90(image, k=3)
    out = model.predict_segmentation(
        inp=im,
        out_fname="output.png"
    )
    t = cv2.imread("output.png")
    cv2.imshow("a", t)
    cv2.imshow("image", image)
    cv2.waitKey(100)
