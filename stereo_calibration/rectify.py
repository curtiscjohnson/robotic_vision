import cv2 as cv
import yaml
import numpy as np

#get one pair of images from stereo dataset
left_img = cv.imread('data/20240207145416/L/0.png')
right_img = cv.imread('data/20240207145416/R/0.png')

#load left and right camera calibration coeffs from yaml
with open('left_calibration.yaml', 'r') as file:
    left_camera_calibration = yaml.load(file, Loader=yaml.FullLoader)

with open('right_calibration.yaml', 'r') as file:
    right_camera_calibration = yaml.load(file, Loader=yaml.FullLoader)

with open('stereo_calibration.yaml', 'r') as file:
    stereo_calibration = yaml.load(file, Loader=yaml.FullLoader)

cameraMatrix1 = left_camera_calibration['camera_matrix']
distCoeffs1 = left_camera_calibration['dist_coeff']
cameraMatrix2 = right_camera_calibration['camera_matrix']
distCoeffs2 = right_camera_calibration['dist_coeff']
R = stereo_calibration['R']
T = stereo_calibration['T']
E = stereo_calibration['E']
F = stereo_calibration['F']

#make all numpy arrays
cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = np.array(distCoeffs1)
cameraMatrix2 = np.array(cameraMatrix2)
distCoeffs2 = np.array(distCoeffs2)
R = np.array(R)
T = np.array(T)
E = np.array(E)
F = np.array(F)

#use stereo rectify to get rectification matrices
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(cameraMatrix1,
                                                 distCoeffs1,
                                                 cameraMatrix2,
                                                 distCoeffs2,
                                                 left_img.shape[:2][::-1],
                                                 R,
                                                 T,
                                                 flags=cv.CALIB_ZERO_DISPARITY,
                                                 alpha=0)

#use initUndistortRectifyMap to get rectification maps
#for left camera
left_mapx, left_mapy = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1,
                                                  R1, P1,
                                                  left_img.shape[:2][::-1],
                                                  cv.CV_32FC1)

#for right camera
right_mapx, right_mapy = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2,
                                                    R2, P2,
                                                    right_img.shape[:2][::-1],
                                                    cv.CV_32FC1)

#apply remap to rectify images
undistorted_left = cv.remap(left_img, left_mapx, left_mapy, cv.INTER_LINEAR)
undistorted_right = cv.remap(right_img, right_mapx, right_mapy,
                             cv.INTER_LINEAR)

abs_diff_left = cv.absdiff(left_img, undistorted_left)
abs_diff_right = cv.absdiff(right_img, undistorted_right)

#draw 3 random horizontal lines at y = 163, 233, 119
cv.line(undistorted_left, (0, 100), (left_img.shape[1], 100), (0, 255, 0), 2)
cv.line(undistorted_left, (0, 200), (left_img.shape[1], 200), (0, 255, 0), 2)
cv.line(undistorted_left, (0, 300), (left_img.shape[1], 300), (0, 255, 0), 2)

cv.line(undistorted_right, (0, 100), (right_img.shape[1], 100), (0, 255, 0), 2)
cv.line(undistorted_right, (0, 200), (right_img.shape[1], 200), (0, 255, 0), 2)
cv.line(undistorted_right, (0, 300), (right_img.shape[1], 300), (0, 255, 0), 2)

#display undistorted images
cv.imshow('left', left_img)
cv.imshow('right', right_img)

cv.imshow('undistorted left', undistorted_left)
cv.imshow('undistorted right', undistorted_right)

cv.imshow('abs_diff_left', abs_diff_left)
cv.imshow('abs_diff_right', abs_diff_right)

cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('original_left.png', left_img)
cv.imwrite('original_right.png', right_img)

cv.imwrite('rectified_left.png', undistorted_left)
cv.imwrite('rectified_right.png', undistorted_right)

cv.imwrite('abs_diff_left.png', abs_diff_left)
cv.imwrite('abs_diff_right.png', abs_diff_right)
