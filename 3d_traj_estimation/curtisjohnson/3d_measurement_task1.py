import cv2 as cv
import numpy as np
import yaml

left_stereo_calib = cv.imread('./stereo_calib/L/0.png')
right_stereo_calib = cv.imread('./stereo_calib/R/0.png')

#use findChessboardCorner to find the corners of the chessboard
left_stereo_calib_gray = cv.cvtColor(left_stereo_calib, cv.COLOR_BGR2GRAY)
right_stereo_calib_gray = cv.cvtColor(right_stereo_calib, cv.COLOR_BGR2GRAY)

ret1, corners1 = cv.findChessboardCorners(left_stereo_calib_gray, (10, 7),
                                          None)
ret2, corners2 = cv.findChessboardCorners(right_stereo_calib_gray, (10, 7),
                                          None)

#refine corners with subCornerPix
winSize = (11, 11)
zeroZone = (-1, -1)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

refined_corners1 = cv.cornerSubPix(left_stereo_calib_gray, corners1, winSize,
                                   zeroZone, criteria)

refined_corners2 = cv.cornerSubPix(right_stereo_calib_gray, corners2, winSize,
                                   zeroZone, criteria)

#draw the corners on the images
# left_stereo_calib = cv.drawChessboardCorners(left_stereo_calib, (10, 7),
#  refined_corners1, ret1)
# cv.imshow('left_stereo_calib', left_stereo_calib)
# cv.waitKey(0)

#load left and right camera parameters from yaml
with open('./calibration/left_calibration.yaml') as f:
    leftdict = yaml.load(f, Loader=yaml.FullLoader)

with open('./calibration/right_calibration.yaml') as f:
    rightdict = yaml.load(f, Loader=yaml.FullLoader)

with open('./calibration/stereo_calibration.yaml') as f:
    stereodict = yaml.load(f, Loader=yaml.FullLoader)

left_camera_matrix = np.array(leftdict['camera_matrix'])
left_distortion = np.array(leftdict['dist_coeff'])
right_camera_matrix = np.array(rightdict['camera_matrix'])
right_distortion = np.array(rightdict['dist_coeff'])

R = np.array(stereodict['R'])
T = np.array(stereodict['T'])
E = np.array(stereodict['E'])
F = np.array(stereodict['F'])

#get 4 outermost corner points, 0, 9, 60, 69 indexes
four_corners1 = np.array([
    refined_corners1[0], refined_corners1[9], refined_corners1[60],
    refined_corners1[69]
])

four_corners2 = np.array([
    refined_corners2[0], refined_corners2[9], refined_corners2[60],
    refined_corners2[69]
])

#draw circles on the 4 outermost corners
for corner in four_corners1:
    left_stereo_calib = cv.circle(left_stereo_calib,
                                  (corner[0][0], corner[0][1]), 5,
                                  (255, 0, 255), -1)

for corner in four_corners2:
    right_stereo_calib = cv.circle(right_stereo_calib,
                                   (corner[0][0], corner[0][1]), 5,
                                   (255, 0, 255), -1)

cv.imshow('left_stereo_calib', left_stereo_calib)
cv.imshow('right_stereo_calib', right_stereo_calib)
cv.imwrite('left_stereo_calib_4corners.png', left_stereo_calib)
cv.imwrite('right_stereo_calib_4corners.png', right_stereo_calib)
cv.waitKey(0)

#get rectification parameters from stereoRectify
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(
    left_camera_matrix, left_distortion, right_camera_matrix, right_distortion,
    left_stereo_calib_gray.shape[::-1], R, T)

#use the parameters to undistort AND rectify the 4 outermost corner points
undistorted1 = cv.undistortPoints(four_corners1,
                                  left_camera_matrix,
                                  left_distortion,
                                  R=R1,
                                  P=P1)
undistorted2 = cv.undistortPoints(four_corners2,
                                  right_camera_matrix,
                                  right_distortion,
                                  R=R2,
                                  P=P2)
print(f"four_corners1:\n {four_corners1}")
print(f"four_corners2:\n {four_corners2}")


print(f"undistorted1:\n {undistorted1}")
print(f"undistorted2:\n {undistorted2}")

#calculate disparity for each point, only differnce x1 - x2
disparity = undistorted1 - undistorted2

#get only x values
disparity = disparity[:, :, 0].squeeze().reshape(4, 1)
print(f"disparity x:\n {disparity}")

undistorted1 = undistorted1.squeeze()
undistorted2 = undistorted2.squeeze()

#! this just needs points in a weird format, matching the weird output of undistortPoints
undistorted1 = np.hstack((undistorted1, disparity)).reshape(4, 1, 3)
undistorted2 = np.hstack((undistorted2, disparity)).reshape(4, 1, 3)

print(f"undistorted1 with disparity:\n {undistorted1}")
print(f"undistorted2 with disparity:\n {undistorted2}")

# use perspectiveTransform to get the 3D coordinates of the 4 outermost corner points
points3d1 = cv.perspectiveTransform(undistorted1, Q).squeeze()
points3d2 = cv.perspectiveTransform(undistorted2, Q).squeeze()
print(f"points3d1:\n {points3d1}")
print(f"points3d2:\n {points3d2}")

#prove it
left_should_be = points3d2 + np.array([np.linalg.norm(T), 0, 0])
print(
    f"Discrepancy between left and right measurement:\n {left_should_be - points3d1}"
)

#actual distance between the 4 outermost corner points
actual_distance = np.linalg.norm(points3d1[0, :] - points3d1[1, :])
print(f"Actual distance between 2 corner points: {actual_distance}")
print(f"Should be {3.88*9}")
