#load calibration data
import yaml
import numpy as np
import cv2 as cv

with open("./webcam_calibration.yaml", "r") as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)

mtx = np.array(loadeddict.get('camera_matrix'))
dist = np.array(loadeddict.get('dist_coeff'))

#load the 3 test images Far, Close, Turned
img = cv.imread('./webcam_calibration_imgs/calibration_img_0.jpg')

# Undistort the images
undistorted_img = cv.undistort(img, mtx, dist, None, mtx)

#compute the absdiff between original and undistorted images
img_diff = cv.absdiff(img, undistorted_img)

#show the results
cv.imshow('far_diff', img_diff)

cv.waitKey(0)

#save diff images
cv.imwrite('./webcam_calibration_imgs/diff.jpg', img_diff)

cv.destroyAllWindows()

