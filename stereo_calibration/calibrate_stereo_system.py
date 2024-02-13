import glob
from re import L
import cv2 as cv
import yaml
import numpy as np


def getChesboardObjPoints(chessboard_size, numImages, square_size):
    objP = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objP[:, :2] = np.mgrid[0:chessboard_size[0],
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objP *= square_size
    objP = [objP] * numImages
    return np.array(objP)


def getChessboardImgPoints(calibration_files):

    img_points = []

    for file in calibration_files:
        img = cv.imread(file)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('gray', gray)

        # Find the chess board initial_corners
        ret, initial_corners = cv.findChessboardCorners(gray, (10, 7), None)

        # If found, add object points, image points
        if ret == True:
            print(f"Found initial_corners in {file}.")
            #refine initial_corners found
            # Set the needed parameters to find the refined corners
            winSize = (11, 11)
            zeroZone = (-1, -1)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30,
                        0.001)

            refined_corners = cv.cornerSubPix(gray, initial_corners, winSize,
                                              zeroZone, criteria)
            img_points.append(refined_corners)

            # Draw and display the initial_corners
            cv.drawChessboardCorners(img, (10, 7), refined_corners, ret)
            cv.imshow('img', img)

            # k = cv.waitKey(0)
            # if k == ord('q'):
            #     break
        else:
            print(f"Could not find initial_corners in {file}.")

    cv.destroyAllWindows()

    return np.array(img_points)


#load left and right camera calibration coeffs from yaml
with open('left_calibration.yaml', 'r') as file:
    left_camera_calibration = yaml.load(file, Loader=yaml.FullLoader)

with open('right_calibration.yaml', 'r') as file:
    right_camera_calibration = yaml.load(file, Loader=yaml.FullLoader)

left_calibration_files = sorted(glob.glob('data/20240207145416/L/*.png'))
right_calibration_files = sorted(glob.glob('data/20240207145416/R/*.png'))

# #load left and right camera calibration coeffs from yaml FOR PRACTICE IMAGES
# with open('left_practice_calibration.yaml', 'r') as file:
#     left_camera_calibration = yaml.load(file, Loader=yaml.FullLoader)

# with open('right_practice_calibration.yaml', 'r') as file:
#     right_camera_calibration = yaml.load(file, Loader=yaml.FullLoader)

# left_calibration_files = sorted(glob.glob('./Practice/SL/*.bmp'))
# right_calibration_files = sorted(glob.glob('./Practice/SR/*.bmp'))

cameraMatrix1 = left_camera_calibration['camera_matrix']
distCoeffs1 = left_camera_calibration['dist_coeff']
cameraMatrix2 = right_camera_calibration['camera_matrix']
distCoeffs2 = right_camera_calibration['dist_coeff']

#make all numpy arrays
cameraMatrix1 = np.array(cameraMatrix1)
distCoeffs1 = np.array(distCoeffs1)
cameraMatrix2 = np.array(cameraMatrix2)
distCoeffs2 = np.array(distCoeffs2)

imageSize = cv.imread(left_calibration_files[0]).shape[:2][::-1]
objectPoints = getChesboardObjPoints(chessboard_size=(10, 7),
                                     numImages=len(left_calibration_files),
                                     square_size=3.88)  #3.88 inches in meters

imagePoints1 = getChessboardImgPoints(left_calibration_files)
imagePoints2 = getChessboardImgPoints(right_calibration_files)

# np.save('objectPoints.npy', objectPoints)
# np.save('imagePoints1.npy', imagePoints1)
# np.save('imagePoints2.npy', imagePoints2)

retval, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
    objectPoints,
    imagePoints1,
    imagePoints2,
    cameraMatrix1,
    distCoeffs1,
    cameraMatrix2,
    distCoeffs2,
    imageSize,
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    flags=cv.CALIB_FIX_INTRINSIC)

np.set_printoptions(suppress=True, precision=4)
print("cameraMatrix1\n", cameraMatrix1)
print("distCoeffs1\n", distCoeffs1)
print("cameraMatrix2\n", cameraMatrix2)
print("distCoeffs2\n", distCoeffs2)

print("R\n", R)
print("T\n", T)


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


print("E\n", E)
print(f"My own calculated E:\n {skew(T.flatten()) @ R}")

# why does this not work? The math doesn't work with the E and F that we got.
print(f"Other E:\n{cameraMatrix2.T @ F @ cameraMatrix1}")

print("F\n", F)

F_mine = np.linalg.inv(cameraMatrix2).T @ E @ np.linalg.inv(cameraMatrix1)
print(f"My own calculated F:\n {F_mine}")

#write all this to a file
stereo_calibration = {
    'R': R.tolist(),
    'T': T.tolist(),
    'E': E.tolist(),
    'F': F.tolist()
}

with open('stereo_calibration.yaml', 'w') as file:
    yaml.dump(stereo_calibration, file)
