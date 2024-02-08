import cv2 as cv
import yaml
import numpy as np

#get one pair of images from stereo dataset
left_img = cv.imread('data/20240207145416/L/0.png')
right_img = cv.imread('data/20240207145416/R/0.png')

# use undistort to remove distortion
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

undistorted_left = cv.undistort(left_img, cameraMatrix1, distCoeffs1)
undistorted_right = cv.undistort(right_img, cameraMatrix2, distCoeffs2)


#display undistorted images
def click_event(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print(f'Pixel coordinates: x = {x}, y = {y}')


cv.namedWindow('left')
cv.setMouseCallback('left', click_event)

cv.namedWindow('right')
cv.setMouseCallback('right', click_event)

cv.namedWindow('undistorted left')
cv.setMouseCallback('undistorted left', click_event)

cv.namedWindow('undistorted right')
cv.setMouseCallback('undistorted right', click_event)

#put images in windows
cv.imshow('left', left_img)
cv.imshow('right', right_img)
cv.imshow('undistorted left', undistorted_left)
cv.imshow('undistorted right', undistorted_right)

cv.imshow('abs_diff_left', cv.absdiff(left_img, undistorted_left))
cv.imshow('abs_diff_right', cv.absdiff(right_img, undistorted_right))

cv.waitKey(0)
cv.destroyAllWindows()

#these points I got from just clicking
left_points = np.array([[336, 163], [362, 233], [461, 119]])
right_points = np.array([[350, 371], [399, 178], [489, 233]])

#draw cirlces on the left and right images for these points
for point in left_points:
    cv.circle(left_img, tuple(point), 2, (0, 0, 255), -1)

for point in right_points:
    cv.circle(right_img, tuple(point), 2, (0, 0, 255), -1)

#compute correspond epipolar lines
lines1 = cv.computeCorrespondEpilines(left_points.reshape(-1, 1, 2), 1, F)
lines1 = lines1.reshape(-1, 3)

lines2 = cv.computeCorrespondEpilines(right_points.reshape(-1, 1, 2), 2, F)
lines2 = lines2.reshape(-1, 3)

#plot lines on the right image
for line in lines1:
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [
        right_img.shape[1], -(line[2] + line[0] * right_img.shape[1]) / line[1]
    ])
    cv.line(right_img, (x0, y0), (x1, y1), (0, 255, 0), 1)

#plot lines on the left image
for line in lines2:
    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [
        left_img.shape[1], -(line[2] + line[0] * left_img.shape[1]) / line[1]
    ])
    cv.line(left_img, (x0, y0), (x1, y1), (0, 255, 0), 1)

cv.imshow('left', left_img)
cv.imshow('right', right_img)

cv.waitKey(0)
cv.destroyAllWindows()
