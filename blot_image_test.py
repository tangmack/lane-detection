import numpy as np
import cv2
import glob

files_list = glob.glob("select_images/*")

for file in files_list:
    img = cv2.imread(file, 0)
    # seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
    # binaryImg = regionGrow(img, seeds, 10)
    cv2.imshow(' ', img)
    cv2.waitKey(0)