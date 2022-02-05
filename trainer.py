import sys

import numpy as np
import cv2

im = cv2.imread('train.png')

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3, 3), 0)
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
(T, thresh) = cv2.threshold(opening, 120, 255, cv2.THRESH_BINARY)

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 200))
responses = []
keys = [i for i in range(48, 58)]

for cnt in contours:
    if cv2.contourArea(cnt) > 70:
        [x, y, w, h] = cv2.boundingRect(cnt)
        # cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
        # cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # cv2.imshow("a", im)
        if 17 < h and w < 30:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = thresh[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (10, 20))
            cv2.imshow('norm', im)
            cv2.imshow("cut", roismall)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, 200))
                samples = np.append(samples, sample, 0)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print("training complete")

np.savetxt('generalsamples.data', samples)
np.savetxt('generalresponses.data', responses)
# cv2.waitKey(0)