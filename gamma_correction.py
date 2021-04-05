import cv2
import numpy as np
import skvideo.io


gamma = 4.5

cap = cv2.VideoCapture('night/Night Drive - 2689.mp4')
# cap = cv2.VideoCapture('data_2/challenge_video.mp4')
# cap = cv2.VideoCapture('data1.mp4')

writer = skvideo.io.FFmpegWriter("NightDriveOutputVideo.mp4")
while (cap.isOpened()):
    ret, frame_original = cap.read()
    frame = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)

    # frame =  cv2.GaussianBlur(gray, (21, 21), 0)
    # th3 = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, -1)
    # ret2, th3 = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # frame = cv2.GaussianBlur(frame_original, (3, 3), 0)

    img_normalized = frame/255
    gamma_inverse = 1/gamma
    gamma_image = img_normalized**(gamma_inverse)

    writer.writeFrame(gamma_image)
    cv2.imshow("gamma_image", gamma_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()