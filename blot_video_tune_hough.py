import numpy as np
import cv2

import numpy as np
from scipy import ndimage

import unwarp_util
import math

def callback(x):
    pass

# video_name = 'night/Night Drive - 2689'
# video_name = 'data_2/challenge_video' # todo input
video_name = 'data1'

if video_name == 'night/Night Drive - 2689':
    string_name = 'Night Drive - 2689'
elif video_name == 'data_2/challenge_video':
    string_name = 'challenge_video'
elif video_name == 'data1':
    string_name = 'data1'

cap = cv2.VideoCapture(video_name + '.mp4')

if cap.isOpened():
    # get cap property
    img_width  = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )   # float `img_width`
    img_height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) ) # float `img_height`
    fps = cap.get(cv2.CAP_PROP_FPS)



if video_name == 'data_2/challenge_video':
    crop_height = int(img_height/2+img_height*.15)
elif video_name == 'data1':
    crop_height = int(img_height/2)

cv2.namedWindow('image')
rho=1
theta=np.pi / 180
threshold=20
minLineLength=0
maxLineGap=0

degree_low = 10
degree_high = 65
# create trackbars for color change
cv2.createTrackbar('rho','image',rho,255,callback)
# cv2.createTrackbar('theta','image',theta,255.0,callback)
cv2.createTrackbar('threshold','image',threshold,255,callback)
cv2.createTrackbar('minLineLength','image',minLineLength,255,callback)
cv2.createTrackbar('maxLineGap','image',maxLineGap,255,callback)

cv2.createTrackbar('degree_low','image',degree_low,100,callback)
cv2.createTrackbar('degree_high','image',degree_high,100,callback)




############### White filter ######################
ilowH = 0
ihighH = 255
ilowS = 0
ihighS = 25
# ilowV = 172 # good for park
ilowV = 190 # good for fremont
ihighV = 255

############### Yellow filter ######################
# ilowH_yellow = 0
# ihighH_yellow = 50
# ilowS_yellow = 80
# # ihighS_yellow = 210
# ihighS_yellow = 135 # drop to 135 to keep greenish yellowish train tracks showing up on park
# ilowV_yellow = 142
# ihighV_yellow = 240

ilowH_yellow = 10
ihighH_yellow = 29
ilowS_yellow = 67
ihighS_yellow = 150 # drop to 135 to keep greenish yellowish train tracks showing up on park
ilowV_yellow = 146
# ihighV_yellow = 135
ihighV_yellow = 232 # up to 240 to account for 227 or so yellow line middle being deleted


count = 0
while(cap.isOpened()):

    rho = cv2.getTrackbarPos('rho', 'image')
    # theta = cv2.getTrackbarPos('theta', 'image')
    threshold = cv2.getTrackbarPos('threshold', 'image')
    minLineLength = cv2.getTrackbarPos('minLineLength', 'image')
    maxLineGap = cv2.getTrackbarPos('maxLineGap', 'image')
    degree_low = cv2.getTrackbarPos('degree_low', 'image')
    degree_high = cv2.getTrackbarPos('degree_high', 'image')


    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    cropped_image = frame[crop_height:-1,0:img_width]


    hsv_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # equalized = cv2.equalizeHist(gray)
    # blur = cv2.GaussianBlur(equalized, (9, 9), 0)

    # ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # th3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -8)
    th4 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -8)

    # mask = cv2.merge((th4, th4, th4)) # merge single channel into 3 channel

    res = cv2.bitwise_and(cropped_image, cropped_image, mask=th4)
    res_hsv = cv2.bitwise_and(hsv_cropped, hsv_cropped, mask=th4)

    # res_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask_hsv = cv2.inRange(res_hsv, lower_hsv, higher_hsv)


    ##### Yellow ######
    lower_hsv_yellow = np.array([ilowH_yellow, ilowS_yellow, ilowV_yellow])
    higher_hsv_yellow = np.array([ihighH_yellow, ihighS_yellow, ihighV_yellow])
    mask_hsv_yellow = cv2.inRange(res_hsv, lower_hsv_yellow, higher_hsv_yellow)
    res_hsv_thresholded = cv2.bitwise_and(res_hsv, res_hsv, mask=mask_hsv_yellow)
    mask_combined_yellow = th4 & mask_hsv_yellow
    ###################

    # mask_combined = th4 & mask_hsv
    mask_combined = (mask_hsv | mask_hsv_yellow) & th4
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)

    res = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_combined)



    # Probabilistic Line Transform
    # dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
    # lines: A vector that will store the parameters (r,θ) of the detected lines
    # rho : The resolution of the parameter r in pixels. We use 1 pixel.
    # theta: The resolution of the parameter θ in radians. We use 1 degree (CV_PI/180)
    # threshold: The minimum number of intersections to "*detect*" a line
    # srn and stn: Default parameters to zero. Check OpenCV reference for more info.
    # linesP = cv2.HoughLinesP(image=mask_hsv, rho=1, theta=np.pi / 180, threshold=20, lines=None, minLineLength=0, maxLineGap=0)
    # lines_var = []
    # linesP = cv2.HoughLinesP(image=mask_hsv, rho=rho, theta=np.pi / 180, threshold=threshold, lines=None, minLineLength=minLineLength, maxLineGap=maxLineGap)
    linesP = cv2.HoughLinesP(image=mask_combined, rho=rho, theta=np.pi / 180, threshold=threshold, lines=None, minLineLength=minLineLength, maxLineGap=maxLineGap)


    hough_only = np.zeros(gray.shape)
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            try:
                angle = int(math.atan((l[1] - l[3]) / (l[0] - l[2])) * 180 / math.pi)
            except:
                angle = 99

            if abs(angle) > degree_low and abs(angle) < degree_high:
                angle_string = str(angle) # theta
                cv2.putText(cropped_image, angle_string, (l[0], l[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                cv2.line(cropped_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(hough_only, (l[0], l[1]), (l[2], l[3]), 255, 3, cv2.LINE_AA) # also draw on blank hough image

            else:
                cv2.putText(cropped_image, ".", (l[0], l[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))


    print(count)
    cv2.imshow('25',res)
    cv2.imshow("hsv_cropped", hsv_cropped)
    cv2.imshow("th4", th4)
    cv2.imshow("mask_hsv", mask_hsv)
    cv2.imshow("mask_combined", mask_combined)
    cv2.imshow("hough_only", hough_only)
    cv2.imshow("image", cropped_image)
    cv2.imshow("mask_combined_yellow", mask_combined_yellow)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    count += 1

cv2.destroyAllWindows()
cap.release()