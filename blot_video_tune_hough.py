import numpy as np
import cv2

import numpy as np
from scipy import ndimage

import unwarp_util
import math
import matplotlib.pyplot as plt

import statistics


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

    # left_b_threshold_lower =
    # left_b_threshold_upper =
    #
    # right_b_threshold_lower =
    # right_b_threshold_upper =

elif video_name == 'data1':
    crop_height = int(img_height/2)

    # left_b_threshold_lower = 330 #
    # left_b_threshold_upper = 490
    #
    # right_b_threshold_lower = -2000
    # right_b_threshold_upper = -630

cv2.namedWindow('image')
rho=1
theta=np.pi / 180
threshold=20
minLineLength=0
maxLineGap=0

degree_low = 27
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

hh = 300
cv2.namedWindow('25', ), cv2.moveWindow('25', 2000,0)
cv2.namedWindow("hsv_cropped", ), cv2.moveWindow("hsv_cropped", 2000,hh)
cv2.namedWindow("th4", ), cv2.moveWindow("th4", 2000,hh*2)
cv2.namedWindow("mask_hsv", ), cv2.moveWindow("mask_hsv", 2000,hh*3)
cv2.namedWindow("hough_only", ), cv2.moveWindow("hough_only", 2000,hh*5)
cv2.namedWindow("mask_combined_yellow", ), cv2.moveWindow("mask_combined_yellow", 2000,hh*4)

cv2.namedWindow('image', ), cv2.moveWindow('image', 2000-img_width-20,hh*2)
cv2.namedWindow("mask_combined", ), cv2.moveWindow("mask_combined", 2000-img_width,hh*4-20)
cv2.namedWindow("im2", ), cv2.moveWindow("im2", 2000-img_width,hh*5-20)
cv2.namedWindow("contours_to_delete_mask", ), cv2.moveWindow("contours_to_delete_mask", 2000-img_width,hh*6-20)
cv2.namedWindow("otsu_threshold", ), cv2.moveWindow("otsu_threshold", 2000-img_width,hh*1-20)



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
    cropped_image_2 = cropped_image.copy()


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
    linesP = cv2.HoughLinesP(image=mask_combined, rho=rho, theta=np.pi / 180, threshold=threshold, lines=None, minLineLength=3, maxLineGap=maxLineGap)

    # lines = cv2.HoughLines(image=mask_combined, rho=1, theta= 1 * np.pi / 180, threshold=35, lines=None, srn=0, stn=0)
    #
    # hough_only = np.zeros(gray.shape)
    # # Draw the lines
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         theta_degrees = theta * 180 / np.pi
    #         if theta_degrees <= 50 or theta_degrees >= 130:
    #             a = math.cos(theta)
    #             b = math.sin(theta)
    #             x0 = a * rho
    #             y0 = b * rho
    #             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #             cv2.putText(cropped_image, str(theta * 180 / np.pi), (pt2), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
    #             cv2.line(hough_only, pt1, pt2, 255, 1, cv2.LINE_AA)
    #             cv2.line(cropped_image, pt1, pt2, (255,255,0), 1, cv2.LINE_AA)


    hough_only = np.zeros(mask_combined.shape)
    # Draw the lines
    positive_angles = []
    negative_angles = []
    positive_angle_b = []
    negative_angle_b = []
    positive_angle_m = []
    negative_angle_m = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]

            try:
                angle = int(math.atan((l[1] - l[3]) / (l[0] - l[2])) * 180 / math.pi)

                m = (l[1] - l[3]) / (l[0] - l[2]) # now try to calculate b

                b = l[1] - m * l[0]
            except:
                angle = 99
                b = 9999
                print("angle exception")

            if abs(angle) > degree_low and abs(angle) < degree_high:
                angle_string = str(angle) # theta
                cv2.putText(cropped_image, angle_string, (l[0], l[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                cv2.line(cropped_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(hough_only, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA) # also draw on blank hough image

                if angle > 0:
                    positive_angles.append(angle)
                    positive_angle_b.append((b))
                    positive_angle_m.append(m)
                elif angle < 0:
                    negative_angles.append(angle)
                    negative_angle_b.append((b))
                    negative_angle_m.append(m)

            else:
                cv2.putText(cropped_image, ".", (l[0], l[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))

    ####################################################################################### need to filter out large white cars
    # otsu_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, -5)
    ret1, otsu_threshold = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    contours_car, hierarchy_car = cv2.findContours(otsu_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    white_cars_mask = np.zeros(mask_combined.shape)
    big_contours = []
    for idx, cnt in enumerate(contours_car):
        area = cv2.contourArea(cnt)
        if area > 12000:
            cv2.drawContours(white_cars_mask, contours_car, idx, 255, cv2.FILLED) # mark for deletion!
            # x, y, w, h = cv2.boundingRect(contours_car[idx])
            # cv2.putText(white_cars_mask, str(round(area)), (x+100,y+100), cv2.FONT_HERSHEY_DUPLEX, 0.5, 80)

    kernel = np.ones((5, 5), np.uint8)
    white_cars_mask = cv2.dilate(white_cars_mask, kernel, iterations=1)
    white_cars_mask_inverted = (255 - white_cars_mask).astype(np.uint8)





















    ######################################################################################## need to filter out Diamond road pattern, wide objects
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(hough_only.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2 = closing.copy()
    # cv2.drawContours(im2, contours, -1, 255, 3)


    contours_to_delete_mask = np.zeros(mask_combined.shape) # zero matrix
    for idx, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        height = rect[1][0]
        width = rect[1][1]
        min_dimension = min(height, width)
        if min_dimension >= 28:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(im2, [box], 0, 80, 2)
            cv2.drawContours(contours_to_delete_mask, contours, idx, 255, cv2.FILLED) # mark for deletion!
            cv2.putText(im2, str(round(min_dimension)), tuple(map(int,rect[0])), cv2.FONT_HERSHEY_DUPLEX, 0.5, 100)
        else:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(im2, [box], 0, 255, 2)
            cv2.putText(im2, str(round(min_dimension)), tuple(map(int,rect[0])), cv2.FONT_HERSHEY_DUPLEX, 0.5, 100)

    # dilate mask a little

    contours_to_delete_mask = cv2.dilate(contours_to_delete_mask, kernel, iterations=1)

    # res = cv2.bitwise_and(cropped_image, cropped_image, mask=mask_combined)
    contours_to_delete_mask_inverted = (255 - contours_to_delete_mask).astype(np.uint8)
    mask_combined_wide_removed = mask_combined & contours_to_delete_mask_inverted & white_cars_mask_inverted



    ############################################################################################################################################################

    linesP_2 = cv2.HoughLinesP(image=mask_combined_wide_removed, rho=rho, theta=np.pi / 180, threshold=threshold, lines=None, minLineLength=3, maxLineGap=maxLineGap)

    hough_only_2 = np.zeros(mask_combined.shape)
    # Draw the lines
    positive_angles = []
    negative_angles = []
    positive_angle_b = []
    negative_angle_b = []
    positive_angle_m = []
    negative_angle_m = []
    if linesP_2 is not None:
        for i in range(0, len(linesP_2)):
            l = linesP_2[i][0]

            try:
                angle = int(math.atan((l[1] - l[3]) / (l[0] - l[2])) * 180 / math.pi)

                m = (l[1] - l[3]) / (l[0] - l[2]) # now try to calculate b

                b = l[1] - m * l[0]
            except:
                angle = 99
                b = 9999
                print("angle exception")

            if abs(angle) > degree_low and abs(angle) < degree_high:
                angle_string = str(angle) # theta
                cv2.putText(cropped_image_2, angle_string, (l[0], l[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, 255)
                cv2.line(cropped_image_2, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(hough_only, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA) # also draw on blank hough image

                if angle > 0:
                    positive_angles.append(angle)
                    positive_angle_b.append((b))
                    positive_angle_m.append(m)
                elif angle < 0:
                    negative_angles.append(angle)
                    negative_angle_b.append((b))
                    negative_angle_m.append(m)

            else:
                cv2.putText(cropped_image_2, ".", (l[0], l[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0))


    ############################################################################################################################################################


    if len(positive_angles) != 0:
        # average_positive = sum(positive_angles) / len(positive_angles) # Average
        # average_positive_b = sum(positive_angle_b) / len(positive_angle_b)
        # average_positive_m = sum(positive_angle_m) / len(positive_angle_m)

        average_positive = statistics.median(positive_angles) # Median
        average_positive_b = statistics.median(positive_angle_b)
        average_positive_m = statistics.median(positive_angle_m)

        cv2.putText(cropped_image_2, str(average_positive), (600,200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))
        cv2.putText(cropped_image_2, str(average_positive_b), (600,250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

        cv2.line(cropped_image_2, (0, int(average_positive_b)), (img_width, int(average_positive_m*img_width+average_positive_b)), (0, 0, 255), 1, cv2.LINE_AA)

    if len(negative_angles) != 0:
        # average_negative = sum(negative_angles) / len(negative_angles) # Average
        # average_negative_b = sum(negative_angle_b) / len(negative_angle_b)
        # average_negative_m = sum(negative_angle_m) / len(negative_angle_m)

        average_negative = statistics.median(negative_angles) # Median
        average_negative_b = statistics.median(negative_angle_b)
        average_negative_m = statistics.median(negative_angle_m)

        cv2.putText(cropped_image_2, str(average_negative), (100,200), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))
        cv2.putText(cropped_image_2, str(average_negative_b), (100,250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255))

        cv2.line(cropped_image_2, (0, int(average_negative_b)), (img_width, int(average_negative_m*img_width+average_negative_b)), (0, 0, 255), 1, cv2.LINE_AA)
    #####################################################################################################################################################################




    # x_plot = positive_angles + negative_angles
    # y_plot = positive_angle_b + negative_angle_b
    # plt.scatter(x_plot,y_plot)
    # plt.show()

    print(count)
    cv2.imshow('25',res)
    cv2.imshow("hsv_cropped", hsv_cropped)
    cv2.imshow("th4", th4)
    cv2.imshow("mask_hsv", mask_hsv)
    cv2.imshow("mask_combined", mask_combined)
    cv2.imshow("hough_only", hough_only)
    cv2.imshow("image", cropped_image_2)
    cv2.imshow("mask_combined_yellow", mask_combined_yellow)
    cv2.imshow("im2", im2)
    cv2.imshow("contours_to_delete_mask", contours_to_delete_mask)
    cv2.imshow("otsu_threshold", white_cars_mask)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    count += 1

cv2.destroyAllWindows()
cap.release()