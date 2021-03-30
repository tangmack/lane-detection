import numpy as np
import cv2

import numpy as np
from scipy import ndimage

# video_name = 'night/Night Drive - 2689'
# video_name = 'data_2/challenge_video'
video_name = 'data1'

if video_name == 'night/Night Drive - 2689':
    string_name = 'Night Drive - 2689'
elif video_name == 'data_2/challenge_video':
    string_name = 'challenge_video'
elif video_name == 'data1':
    string_name = 'data1'

cap = cv2.VideoCapture(video_name + '.mp4')

count = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    if count == 218:
        cv2.imwrite('select_images/' + string_name + '-' + str(count) + '.png', frame)
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # # blur = cv2.GaussianBlur(gray, (51, 51), 0)
    # # blur = cv2.medianBlur(gray,15)
    # equalized = cv2.equalizeHist(gray)
    # # blur = cv2.GaussianBlur(equalized, (9, 9), 0)
    # # blur = cv2.medianBlur(equalized,15)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count +=1

cv2.destroyAllWindows()
cap.release()