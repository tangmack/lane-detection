import cv2
import numpy as np
import glob


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('data1' + '.mp4', 0x7634706d , 30.0, (1392,512))

files_list = glob.glob("data_1/data/*")
for file in files_list:
    # img = cv2.imread('data_1/data/'+'0000000000.png')
    img = cv2.imread(file)
    cv2.imshow("img",img)
    out.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
out.release()