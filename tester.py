import os

import cv2
import numpy as np

samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
filepath = "C:/Users/tommy/PycharmProjects/heartbeat tracker/bae/"
numbers = []
counter = []
pictureNum = 1
for image in os.listdir(filepath):
    tempNum = []
    tempX = []
    im = cv2.imread(filepath + str(image))
    im = cv2.resize(im, (72, 72), interpolation=cv2.INTER_AREA)
    im = im[20:50, :]
    out = np.zeros(im.shape, np.uint8)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(blur, kernel, cv2.BORDER_REFLECT)
    (T, thresh) = cv2.threshold(erode, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 17 and 7 < w < 30:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 20))
            roismall = roismall.reshape((1, 200))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
            string = str(int((results[0][0])))
            tempNum.append(string)
            tempX.append(x)
    zipped_lists = zip(tempX, tempNum)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list = [element for _, element in sorted_zipped_lists]
    number = ""
    for num in sorted_list:
        number = number + num
    print(image + " " + str(number))
    numbers.append(int(number))
    counter.append(pictureNum)
    pictureNum += 1
output = np.array([counter, numbers])
np.savetxt("data.csv", output, delimiter=",")
cv2.waitKey(0)
