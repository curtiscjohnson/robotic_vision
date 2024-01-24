import cv2 as cv
import numpy as np


#read in calibration images from zip folder
def extract_calibration_images(zip_file):
    """Extract calibration images from zip file

    Args:
        zip_file (str): path to zip file containing calibration images

    Returns:
        list: list of calibration images
    """
    import zipfile
    import glob

    #read in calibration images from zip folder
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall("./calibration_imgs/")
    zip_ref.close()

    return glob.glob('./calibration_imgs/*.jpg')


calibration_files = extract_calibration_images('./Calibration Images JPG.zip')

img_points = []
obj_points = []
objP = np.zeros((7 * 10, 3), np.float32)
objP[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)
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
        criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)

        refined_corners = cv.cornerSubPix(gray, initial_corners, winSize,
                                          zeroZone, criteria)
        img_points.append(refined_corners)
        obj_points.append(objP)

        # Draw and display the initial_corners
        cv.drawChessboardCorners(img, (10, 7), refined_corners, ret)
        cv.imshow('img', img)

        # k = cv.waitKey(0)
        # if k == ord('q'):
        # break
    else:
        print(f"Could not find initial_corners in {file}.")

cv.destroyAllWindows()

#calibrate camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points,
                                                  gray.shape[::-1], None, None)
print(f"Camera matrix:\n {mtx}")
print(f"Distortion coefficients:\n {dist}")

sx = 1 / 7.4e-3
focal_length_mm_x = mtx[0, 0] / sx

sy = 1 / 7.4e-3
focal_length_mm_y = mtx[1, 1] / sy

print(f"Focal length in x direction: {focal_length_mm_x} mm")
print(f"Focal length in y direction: {focal_length_mm_y} mm")

#save camera matrix and distortion coefficients to yaml file
import yaml
data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
with open("calibration.yaml", "w") as f:
    yaml.dump(data, f)


