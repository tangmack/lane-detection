import cv2
import numpy as np
import glob
import pickle

def computeMaxWidthMaxHeightList(rect):
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	# ...and now for the height of our new image
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	# take the maximum of the width and height values to reach
	# our final dimensions
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))
	return (maxWidth, maxHeight)

def unWarp(img, corners_list):
	# now that we have our rectangle of points, let's compute
	# the width of our new image
	rect = np.zeros((4, 2), dtype = "float32")
	# rect[0] = [571,499] # top left
	# rect[1] = [739,499] # top right
	# rect[2] = [960,676] # bottom right
	# rect[3] = [297,676] # bottom left


	rect[0] = corners_list[0]  # top left
	rect[1] = corners_list[1]  # top right
	rect[2] = corners_list[2]  # bottom right
	rect[3] = corners_list[3]  # bottom left

	# construct our destination points which will be used to
	# map the screen to a top-down, "birds eye" view
	# a = [rect[3][0], rect[0][1]]
	# b = [rect[2][0], rect[1][1]]
	# c = [rect[2][0], rect[2][1]]
	# d = [rect[3][0], rect[3][1]]
	# dst = np.array([
	# 	[rect[3][0], rect[0][1]],
	# 	[rect[2][0], rect[1][1]],
	# 	[rect[2][0], rect[2][1]],
	# 	[rect[3][0], rect[3][1]]], dtype = "float32")

	maxWidth, maxHeight = computeMaxWidthMaxHeightList(rect)

	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	H = cv2.getPerspectiveTransform(rect, dst)

	warp = cv2.warpPerspective(img, H, (maxWidth, maxHeight))

	return warp

if __name__ == '__main__':

	files_list = glob.glob("select_images/*")
	file = files_list[4]
	print(file)
	img = cv2.imread(file, 1)
	# cv2.imshow('Pick four corners, Top left, Top right, Bottom right, Bottom left', img)
	# cv2.waitKey(0)





	corners_file = 'data1-218.png-corners'
	with open (corners_file, 'rb') as fp:
		corners_list = pickle.load(fp)

	# rect[0] = [560,308] # top left
	# rect[1] = [726,308] # top right
	# rect[2] = [843,500] # bottom right
	# rect[3] = [277,500] # bottom left

	# H = computeHomography(corners_list)
	warp = unWarp(img, corners_list)

	cv2.imshow('unwarped', warp)
	cv2.waitKey(0)













