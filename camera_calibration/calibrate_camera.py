import cv2 as cv
import numpy as np

numFramesToCalibrate = 20
chess_board_size = (10, 8)

img_points = []
obj_points = []
objP = np.zeros(((chess_board_size[0] - 1) * (chess_board_size[1] - 1), 3),
                np.float32)
objP[:, :2] = np.mgrid[0:chess_board_size[0] - 1,
                       0:chess_board_size[1] - 1].T.reshape(-1, 2)

#open webcam
webcam = cv.VideoCapture(0)

while len(img_points) < numFramesToCalibrate:
    img = webcam.read()[1]
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)

    key = cv.waitKey(0)

    if key == ord('q'):
        break
    elif key == ord('s'):
        import os

        # Check if directory exists
        if not os.path.exists("./webcam_calibration_imgs"):
            # If not, create it
            os.makedirs("./webcam_calibration_imgs")

        print(f"Saving image {len(img_points)}")
        cv.imwrite(
            f'./webcam_calibration_imgs/calibration_img_{len(img_points)}.jpg',
            img)

        # Find the chess board initial_corners
        ret, initial_corners = cv.findChessboardCorners(
            gray, (chess_board_size[0] - 1, chess_board_size[1] - 1), None)

        # If found, add object points, image points
        if ret == True:
            print(f"Found initial_corners.")
            #refine initial_corners found
            # Set the needed parameters to find the refined corners
            winSize = (11, 11)
            zeroZone = (-1, -1)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40,
                        0.001)

            refined_corners = cv.cornerSubPix(gray, initial_corners, winSize,
                                              zeroZone, criteria)
            img_points.append(refined_corners)
            obj_points.append(objP)

            # Draw and display the initial_corners
            cv.drawChessboardCorners(
                img, (chess_board_size[0] - 1, chess_board_size[1] - 1),
                refined_corners, ret)
            cv.imshow('img', img)

            # k = cv.waitKey(0)
            # if k == ord('q'):
            # break
        else:
            print(f"Could not find initial_corners.")

cv.destroyAllWindows()

#calibrate camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points,
                                                  gray.shape[::-1], None, None)
print(f"Camera matrix:\n {mtx}")
print(f"Distortion coefficients:\n {dist}")

#save camera matrix and distortion coefficients to yaml file
import yaml

data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
with open("webcam_calibration.yaml", "w") as f:
    yaml.dump(data, f)
