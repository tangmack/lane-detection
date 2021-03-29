import numpy as np
import cv2

import numpy as np
from scipy import ndimage

cap = cv2.VideoCapture('night/Night Drive - 2689.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (51, 51), 0)
    # blur = cv2.medianBlur(gray,15)
    equalized = cv2.equalizeHist(gray)
    # blur = cv2.GaussianBlur(equalized, (9, 9), 0)
    # blur = cv2.medianBlur(equalized,15)

    cv2.imshow('frame',equalized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()