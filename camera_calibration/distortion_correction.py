#load calibration data
import yaml
import numpy as np
import cv2 as cv

with open("./calibration.yaml", "r") as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)

mtx = np.array(loadeddict.get('camera_matrix'))
dist = np.array(loadeddict.get('dist_coeff'))

#load the 3 test images Far, Close, Turned
far_img = cv.imread('./calibration_imgs/Far.jpg')
close_img = cv.imread('./calibration_imgs/Close.jpg')
turned_img = cv.imread('./calibration_imgs/Turn.jpg')

# Undistort the images
undistorted_far_img = cv.undistort(far_img, mtx, dist, None, mtx)
undistorted_close_img = cv.undistort(close_img, mtx, dist, None, mtx)
undistorted_turned_img = cv.undistort(turned_img, mtx, dist, None, mtx)

#compute the absdiff between original and undistorted images
far_diff = cv.absdiff(far_img, undistorted_far_img)
close_diff = cv.absdiff(close_img, undistorted_close_img)
turned_diff = cv.absdiff(turned_img, undistorted_turned_img)

#show the results
cv.imshow('far_diff', far_diff)
cv.imshow('close_diff', close_diff)
cv.imshow('turned_diff', turned_diff)

cv.waitKey(0)

#save diff images
cv.imwrite('./calibration_imgs/far_diff.jpg', far_diff)
cv.imwrite('./calibration_imgs/close_diff.jpg', close_diff)
cv.imwrite('./calibration_imgs/turned_diff.jpg', turned_diff)

cv.destroyAllWindows()

