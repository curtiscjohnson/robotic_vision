#load left and right camera parameters from yaml
import glob
import os
import yaml
import cv2 as cv

import numpy as np

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

left_images = sorted(
    glob.glob('./20240215112959/L/*.png'),
    key=lambda filename: int(os.path.splitext(os.path.basename(filename))[0]))

img = cv.imread(left_images[0])
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

#get rectification parameters from stereoRectify
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(left_camera_matrix,
                                                 left_distortion,
                                                 right_camera_matrix,
                                                 right_distortion,
                                                 img.shape[::-1], R, T)

undistortRectifyMapLx, undistortRectifyMapLy = cv.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, img.shape[::-1], cv.CV_32FC1)

undistortRectifyMapRx, undistortRectifyMapRy = cv.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, img.shape[::-1],
    cv.CV_32FC1)

# #save undistoryRectifyMap to yaml
# with open('./calibration/undistortRectifyMapL.yaml', 'w') as f:
#     yaml.dump(
#         {
#             'undistortRectifyMapLx': undistortRectifyMapLx.tolist(),
#             'undistortRectifyMapLy': undistortRectifyMapLy.tolist()
#         }, f)

# with open('./calibration/undistortRectifyMapR.yaml', 'w') as f:
#     yaml.dump(
#         {
#             'undistortRectifyMapRx': undistortRectifyMapRx.tolist(),
#             'undistortRectifyMapRy': undistortRectifyMapRy.tolist()
#         }, f)

np.save('./calibration/undistortRectifyMapLx.npy', undistortRectifyMapLx)
np.save('./calibration/undistortRectifyMapLy.npy', undistortRectifyMapLy)
np.save('./calibration/undistortRectifyMapRx.npy', undistortRectifyMapRx)
np.save('./calibration/undistortRectifyMapRy.npy', undistortRectifyMapRy)
np.save('./calibration/Q.npy', Q)
np.save('./calibration/R1.npy', R1)
np.save('./calibration/R2.npy', R2)
np.save('./calibration/P1.npy', P1)
np.save('./calibration/P2.npy', P2)

print("undistoryRectify Map saved.")
