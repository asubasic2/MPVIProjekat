import numpy as np
import cv2
import imutils
import copy
from skimage import transform
from skimage import exposure
from skimage import io
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture('Video/DozvoljeniSmjerovi1.mp4')
model = load_model(r"C:\Users\pc\MPVI\Detekcija klasifikatorom\output\istrenirano.model")

labelNames = open("signnames.csv", encoding='utf8').read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]


def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized


def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3, 3), 0)  # parameter
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)  # parameter
    LoG_image = cv2.convertScaleAbs(LoG_image)

    return LoG_image


def binarization(image):
    thresh = cv2.threshold(image, 60, 255, cv2.THRESH_BINARY)[1]
    return thresh


while cap.isOpened():
    ret, image = cap.read()

    if not ret:
        break
    if image is None:
        break
    image = np.rot90(image, k=3)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    cv2.imshow("img", image)
    cv2.waitKey(10)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110, 110, 60])
    upper_blue = np.array([130, 255, 255])

    lower_red = np.array([0, 110, 80])
    upper_red = np.array([4, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([165, 110, 80])
    upper_red = np.array([180, 255, 255])

    lower_yellow = np.array([20, 110, 90])
    upper_yellow = np.array([26, 220, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask3 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = mask0 + mask1 + mask2 + mask3
    resized = cv2.bitwise_and(resized, resized, mask=mask)
    gray = constrastLimit(resized)
    blurred = LaplacianOfGaussian(gray)
    thresh = binarization(blurred)
    cv2.imshow("test", thresh)
    cv2.waitKey(10)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue

        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        area = cv2.contourArea(c)
        if area < 175:
            continue


        c = c.astype("float")
        c *= ratio
        c = c.astype("int")

        x, y, w, h = cv2.boundingRect(c)

        test = image[y - 12:y + h + 12, x - 12:x + w + 12]
        cv2.imwrite("cap.jpg", test)
        test = io.imread("cap.jpg")
        test = transform.resize(test, (32, 32))
        test = exposure.equalize_adapthist(test, clip_limit=0.1)

        test = test.astype("float32") / 255.0
        test = np.expand_dims(test, axis=0)


        preds = model.predict(test)
        j = preds.argmax(axis=1)[0]
        print("Predicted: " + str(j))
        print("Vjerovatnoca: " + str(preds[0][j]))
        if preds[0][j] < 0.7:
            break
        tt = copy.deepcopy(image)
        img = cv2.rectangle(tt, (x - 12, y - 12), (x + w + 12, y + h + 12), (0, 255, 0), 3)
        cv2.putText(img, labelNames[j], (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 160), 1)
        cv2.imshow('frame', img)
        cv2.waitKey(10)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
