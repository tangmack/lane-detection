import cv2
import numpy as np
import glob
import pickle
import os


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(0,255,0),-1)
        mouseX,mouseY = x,y

save_mode = True # todo input here
files_list = glob.glob("select_images/*")
# file = files_list[4] # data1-218
file = files_list[1] # challenge_video-391
print(file)

out_file_name_full = os.path.split(file)
out_file_name = out_file_name_full[1]
corners_file = out_file_name + '-corners' # todo input here
if save_mode == True:

    img = cv2.imread(file, 1)
    cv2.imshow('image', img)

    # img = np.zeros((512,512,3), np.uint8)
    # cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    # corners = [[598, 274], [714, 277], [889, 496], [196, 498]]

    corner_index = 0
    corners = []
    while(corner_index<4):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'): # if 'q' key pressed
            break
        elif k == ord('a'):
            print(mouseX, mouseY)
        elif k == ord('s'):
            corners.append([mouseX, mouseY])
            corner_index += 1

    print(corners)
    for idx, corner in enumerate(corners): # top left and bottom right stay
        if idx == 1:
            corners[idx][1] = corners[0][1] # match y to idx 0
        elif idx == 3:
            corners[idx][1] = corners[2][1] # match y to idx 2

    print(corners)


    with open(corners_file, 'wb') as fp:
        pickle.dump(corners, fp)



else: # show mode
    # corners_file = 'data1-218.png-corners-wider'
    with open(corners_file, 'rb') as fp:
        corners_list = pickle.load(fp)

    img = cv2.imread(file, 1)

    colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0)]
    for idx, corner in enumerate(corners_list):
        cv2.circle(img, (corner[0], corner[1]), 10, colors[idx%4], -1)

    cv2.imshow('image', img)
    cv2.waitKey(0)