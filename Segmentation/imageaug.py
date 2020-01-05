import glob
import ntpath
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import json

#from testimg import path_leaf

seq = iaa.Sequential([
    iaa.Crop(px=(0, 40)),  # crop images from each side by 0 to 50px (randomly chosen)
    iaa.Fliplr(0.6),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 4.0))  # blur images with a sigma of 0 to 4.0
])


def augment_seg(img, seg):
    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def augmentiraj(path, pathseg):
    i = 1000
    while i < 1500:
        for filename in glob.glob(path + '/*.png'):
            end = path_leaf(filename)
            end_json = end[:len(end) - 4]
            imseg = cv2.imread(pathseg + "/" + end)
            im = cv2.imread(filename)
            a, b = augment_seg(im, imseg)
            cv2.imwrite("Dataset/Augmented/" + str(i) + ".png", a)
            cv2.imwrite("Dataset/Augmented_seg/" + str(i) + ".png", b)

            with open('Dataset/Annotations/' + end_json + ".json") as json_file:
                data = json.load(json_file)
                with open("Dataset/Augmented_json/" + str(i) + ".json", 'x') as outfile:
                    json.dump(data, outfile)
            i += 1
    print("I je: " + str(i))


augmentiraj("Dataset/Znakovi_train", "Dataset/Znakovi_segmentation_train")
