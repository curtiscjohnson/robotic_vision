import numpy as np
import yaml
import cv2 as cv

from detect_baseball import BaseballDetector


class Estimator3D:
    def __init__(self):
        #load parameters from file
        self.undistortRectifyMapLx = np.load(
            './calibration/undistortRectifyMapLx.npy')
        self.undistortRectifyMapLy = np.load(
            './calibration/undistortRectifyMapLy.npy')
        self.undistortRectifyMapRx = np.load(
            './calibration/undistortRectifyMapRx.npy')
        self.undistortRectifyMapRy = np.load(
            './calibration/undistortRectifyMapRy.npy')
        self.Q = np.load('./calibration/Q.npy')

        with open('./calibration/stereo_calibration.yaml') as f:
            stereodict = yaml.load(f, Loader=yaml.FullLoader)

        self.R = np.array(stereodict['R'])
        self.T = np.array(stereodict['T'])

        self.left_baseball_detector = BaseballDetector(grayscale=True,
                                                       display=False)
        self.right_baseball_detector = BaseballDetector(grayscale=True,
                                                        display=False)

        self.ball_loc_history = []

    def _undistort_rectify_stereo_images(self, left_img, right_img):
        left_img_rectified = cv.remap(left_img, self.undistortRectifyMapLx,
                                      self.undistortRectifyMapLy,
                                      cv.INTER_LINEAR)
        right_img_rectified = cv.remap(right_img, self.undistortRectifyMapRx,
                                       self.undistortRectifyMapRy,
                                       cv.INTER_LINEAR)

        return left_img_rectified, right_img_rectified

    def _find_ball_pixel_coords(self, left_img_rectified, right_img_rectified):
        left_x, left_y = self.left_baseball_detector.detect(left_img_rectified)
        right_x, right_y = self.right_baseball_detector.detect(
            right_img_rectified)

        return left_x, left_y, right_x, right_y

    def _get_3d_point(self, left_x, left_y, right_x):

        disparity = left_x - right_x

        baseball_center = np.array([left_x, left_y,
                                    disparity]).reshape(1, 1, 3)

        #use Q to get 3D point
        points3d = cv.perspectiveTransform(baseball_center.astype(np.float32),
                                           self.Q)

        return points3d.squeeze()

    def _fit_linear_trajectory(self):
        # x data will be z data, y data will be x data
        # then find x when z is 0
        x = np.array([point[2] for point in self.ball_loc_history])
        y = np.array([point[0] for point in self.ball_loc_history])

        #fit a line to the data
        line_coeffs = np.polyfit(x, y, 1)

        return line_coeffs[0] * 0 + line_coeffs[1]

    def _fit_parabolic_trajectory(self):
        # x data will be z data, y data will be y data
        # then find y when z is 0
        x = np.array([point[2] for point in self.ball_loc_history])
        y = np.array([point[1] for point in self.ball_loc_history])

        #fit a parabola to the data
        parabola_coeffs = np.polyfit(x, y, 2)

        return parabola_coeffs[0] * 0**2 + parabola_coeffs[
            1] * 0 + parabola_coeffs[2]

    def estimate_intercept(self, left_img, right_img):
        #check for greyscale and convert if needed
        if left_img.shape[-1] != 1:
            left_img = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
        if right_img.shape[-1] != 1:
            right_img = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

        left_rect, right_rect = self._undistort_rectify_stereo_images(
            left_img, right_img)

        left_x, left_y, right_x, right_y = self._find_ball_pixel_coords(
            left_rect, right_rect)

        x_intercept = 0
        y_intercept = 0

        if left_x is not None and right_x is not None:
            ball_location = self._get_3d_point(left_x, left_y, right_x)
            self.ball_loc_history.append(ball_location)

            if self.numPointsAcquired >= 2:
                x_intercept = self._fit_linear_trajectory()

            if self.numPointsAcquired >= 3:
                y_intercept = self._fit_parabolic_trajectory()
        return x_intercept, y_intercept
