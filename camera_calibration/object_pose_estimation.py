#load data points from text file
import numpy as np
import cv2 as cv

pixel_points = np.loadtxt('./datapoints.txt',
                          delimiter=' ',
                          dtype=np.float32,
                          usecols=(0, 1),
                          max_rows=20)

inch_points = np.loadtxt('./datapoints.txt',
                         delimiter=' ',
                         dtype=np.float32,
                         usecols=(0, 1, 2),
                         skiprows=20)

#load camera matrix and distortion coefficients from file
import yaml
with open('./calibration.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    camera_matrix = np.asarray(loadeddict.get('camera_matrix'))
    dist_coeffs = np.asarray(loadeddict.get('dist_coeff'))

#solvePnp
retval, rvec, tvec = cv.solvePnP(inch_points, pixel_points, camera_matrix,
                                 dist_coeffs)

rotM, _ = cv.Rodrigues(rvec)

#? what is the translation relative to? The origin of the camera dictated by how we defined the inch_points?
if retval:
    print(f"rotM:\n{rotM}")
    print(f"tvec:\n{tvec}")
else:
    print("solvePnP failed")
