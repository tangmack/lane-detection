import numpy as np
import cv2

import numpy as np
from scipy import ndimage

import pickle

import unwarp_util

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark

# video_name = 'night/Night Drive - 2689'
video_name = 'data_2/challenge_video'
# video_name = 'data1'q

if video_name == 'night/Night Drive - 2689':
    string_name = 'Night Drive - 2689'
elif video_name == 'data_2/challenge_video':
    string_name = 'challenge_video'
elif video_name == 'data1':
    string_name = 'data1'

# cap = cv2.VideoCapture('night/Night Drive - 2689.mp4')
# cap = cv2.VideoCapture('data_2/challenge_video.mp4')
# cap = cv2.VideoCapture('data1.mp4')
cap = cv2.VideoCapture(video_name + '.mp4')

if video_name == 'night/Night Drive - 2689':
    string_name = 'challenge_video-391.png-corners'
elif video_name == 'data_2/challenge_video':
    corners_file = 'challenge_video-391.png-corners'
elif video_name == 'data1':
    corners_file = 'data1-218.png-corners-wider'

with open(corners_file, 'rb') as fp:
    corners_list = pickle.load(fp)

if cap.isOpened():
    # get cap property
    img_width  = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )   # float `img_width`
    img_height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) # float `img_height`
    # or
    # img_width  = cap.get(3)  # float `img_width`
    # img_height = cap.get(4)  # float `img_height`

    fps = cap.get(cv2.CAP_PROP_FPS)


# crop_height = int(img_height/2) + int(img_height*.15)
crop_height = int(img_height/2)

grid_spacing_row = 50
grid_spacing_col = 50
# seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
# seeds = []
# for j in range(0,img_width,grid_spacing_col):
#     for i in range(0,img_height-crop_height,grid_spacing_row):
#         seeds.append(Point(i,j))

# for j in range(0,img_width,grid_spacing_col):
#     for i in range(0,img_height,grid_spacing_row):
#         seeds.append(Point(i,j))


count = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    unwarp = unwarp_util.unWarp(frame,corners_list)

    # cropped_image = frame[crop_height:-1,0:img_width]
    gray = cv2.cvtColor(unwarp, cv2.COLOR_BGR2GRAY)
    # equalized = cv2.equalizeHist(gray)
    # blur = cv2.GaussianBlur(equalized, (9, 9), 0)

    # ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -8)
    th4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -8) # good
    # th4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -4)

    # mask = cv2.merge((th4, th4, th4)) # merge single channel into 3 channel

    res = cv2.bitwise_and(unwarp, unwarp, mask=th4)

    # binaryImg = regionGrow(gray, seeds, 10)
    # cv2.imshow(' ', binaryImg)
    # cv2.waitKey(0)
    # mask original color image with binary image
    # cv2.rectangle(frame, (0, 0), (img_width,int(img_height/2) + int(img_height*.15)), (0, 255, 0), 3)
    # cv2.imshow('11',th3)

    print(count)
    cv2.imshow('25',th4)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    count += 1

cv2.destroyAllWindows()
cap.release()