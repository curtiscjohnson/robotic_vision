from calendar import c
import os
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import glob

import yaml
import matplotlib.pyplot as plt


class BaseballDetector:
    def __init__(self,
                 image_height=480,
                 image_width=640,
                 grayscale=True,
                 display=True) -> None:
        self.prev_image = np.zeros((image_height, image_width), dtype=np.uint8)
        self.incoming_in_gray = grayscale
        self.display = display
        self.plot_img = None
        self.x_bound = 0
        self.y_bound = 0
        self.w_bound = 0
        self.h_bound = 0
        self.first_detect = False
        self.background = np.zeros((image_height, image_width), dtype=np.uint8)

        params = cv.SimpleBlobDetector_Params()
        params.filterByCircularity = True
        params.minCircularity = 0.8
        self.save_counter = 0

        self.blob_detector = cv.SimpleBlobDetector_create(params)

    def _get_ROI(self, img):
        # img comes in with background removed

        #  gaussian blur diff image
        blur = cv.medianBlur(img, 5)

        #use time history to only look in region of interest so that thresholds can be low.
        time_diff = cv.absdiff(blur, self.prev_image)

        #threshold blur on about gresyscale of 5ish
        ret, thresh = cv.threshold(time_diff, 15, 255, cv.THRESH_BINARY)

        # erosion to remove noise and fill in gaps
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv.dilate(thresh, kernel, iterations=3)
        eroded = cv.erode(dilated, kernel, iterations=3)

        #find contour with largest bounding box
        contours, hierarchy = cv.findContours(eroded, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)
        #get max area bounding box
        max_area_idx = 0
        if len(contours) > 0:
            for i, contour in enumerate(contours):
                area = cv.contourArea(contour)
                if area > cv.contourArea(contours[max_area_idx]):
                    max_area_idx = i
            #draw bounding box
            self.x_bound, self.y_bound, self.w_bound, self.h_bound = cv.boundingRect(
                contours[max_area_idx])
            self.w_bound = int(self.w_bound * 1.5)
            self.h_bound = int(self.h_bound * 1.5)

            # Define the text and position
            text = "ROI"
            position = (self.x_bound, self.y_bound - 5
                        )  # Position the text above the rectangle

            # Add the text
            cv.putText(self.plot_img, text, position, cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
            cv.rectangle(
                self.plot_img, (self.x_bound, self.y_bound),
                (self.x_bound + self.w_bound, self.y_bound + self.h_bound),
                (255, 0, 0), 2)

            #crop everything inside of bounding box
            crop_img = blur[self.y_bound:self.y_bound + self.h_bound,
                            self.x_bound:self.x_bound + self.w_bound]
        else:
            crop_img = None

        #threshold on cropped image
        ret, crop_img = cv.threshold(crop_img, 7, 255, cv.THRESH_BINARY)

        self.prev_image = blur

        return blur, thresh, eroded, crop_img, time_diff

    def _get_hough_circle(self, img):

        if img is not None:
            #find circles
            circles = cv.HoughCircles(img,
                                      cv.HOUGH_GRADIENT,
                                      1,
                                      1000,
                                      param1=250,
                                      param2=1,
                                      minRadius=5,
                                      maxRadius=25)
            #
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (circ_x, circ_y, circ_r) in circles:
                    x = self.x_bound + circ_x
                    y = self.y_bound + circ_y
                    #onyl plot circles that are within the bounding box
                    # cv.circle(self.plot_img, (x, y), circ_r, (0, 255, 0), 1)
                    cv.rectangle(self.plot_img, (x - 2, y - 2), (x + 2, y + 2),
                                 (0, 0, 255), -1)
                return (x, y)

        return None, None

    def _get_blob_circle(self, img):
        raise NotImplementedError

    def _remove_background(self, img):
        return cv.absdiff(img, self.background)

    def detect(self, img, display=False):
        self.plot_img = img.copy()
        if not self.incoming_in_gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if not self.first_detect:
            self.background = img.copy()
            self.first_detect = True

        img = self._remove_background(img)

        blur, thresh, eroded, crop_img, time_diff = self._get_ROI(img)
        x_loc, y_loc = self._get_hough_circle(crop_img)

        if self.display:
            cv.imshow('raw', self.plot_img)
            # cv.imshow('diff', diff)
            # cv.imshow('blur', blur)
            # cv.imshow('thresh', thresh)
            # cv.imshow('dilated and eroded', eroded)
            if crop_img is not None:
                cv.imshow('cropped', crop_img)
            # cv.imshow('time diff', time_diff)
            # cv.imshow('background', self.background)
            # cv.imshow('circle', circle)
            key = cv.waitKey(0)

            if key == ord('q'):
                cv.destroyAllWindows()
                raise SystemExit
            elif key == ord('s'):
                cv.imwrite(f'{self.save_counter}.png', self.plot_img)
                self.save_counter += 1

        return x_loc, y_loc


if __name__ == '__main__':

    #load parameters from file
    undistortRectifyMapLx = np.load('./calibration/undistortRectifyMapLx.npy')
    undistortRectifyMapLy = np.load('./calibration/undistortRectifyMapLy.npy')
    undistortRectifyMapRx = np.load('./calibration/undistortRectifyMapRx.npy')
    undistortRectifyMapRy = np.load('./calibration/undistortRectifyMapRy.npy')
    Q = np.load('./calibration/Q.npy')

    with open('./calibration/stereo_calibration.yaml') as f:
        stereodict = yaml.load(f, Loader=yaml.FullLoader)

    R = np.array(stereodict['R'])
    T = np.array(stereodict['T'])

    left_images = sorted(glob.glob('./20240215112959/L/*.png'),
                         key=lambda filename: int(
                             os.path.splitext(os.path.basename(filename))[0]))

    right_images = sorted(glob.glob('./20240215112959/R/*.png'),
                          key=lambda filename: int(
                              os.path.splitext(os.path.basename(filename))[0]))

    left_baseball_detector = BaseballDetector(grayscale=True, display=False)
    right_baseball_detector = BaseballDetector(grayscale=True, display=False)

    ball_traj = []
    for left_file, right_file in zip(left_images, right_images):
        left_img = cv.imread(left_file)
        right_img = cv.imread(right_file)

        left_img_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
        right_img_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

        #undistory and rectify images
        left_img_rect = cv.remap(left_img_gray, undistortRectifyMapLx,
                                 undistortRectifyMapLy, cv.INTER_LINEAR)
        right_img_rect = cv.remap(right_img_gray, undistortRectifyMapRx,
                                  undistortRectifyMapRy, cv.INTER_LINEAR)

        # detect baseball in undisorted and rectified images
        #? could do this, or detect in raw images and only undistort and rectify the ball location. not sure which is faster.
        left_x, left_y = left_baseball_detector.detect(left_img_rect,
                                                       display=False)
        right_x, right_y = right_baseball_detector.detect(right_img_rect,
                                                          display=False)

        print(f'Left: {left_x}, {left_y}')
        print(f'Right: {right_x}, {right_y}')

        if left_x is not None and right_x is not None:
            """ 
            ! To get disparity map, I am individually detecting the ball in 
            ! each image, then using diff in x to get disparity. 
            ! They are not always on the same horizonal line however.
            """

            #calculate estimated disparity for the baseball center
            disparity = left_x - right_x

            baseball_center = np.array([left_x, left_y,
                                        disparity]).reshape(1, 1, 3)

            #use Q to get 3D point
            points3d = cv.perspectiveTransform(
                baseball_center.astype(np.float32), Q)

            print(f'3D point: {points3d}')

            ball_traj.append(points3d.squeeze())

    ball_traj = np.array(ball_traj)

    #shift data to midpoint between cameras. Since this is ambiguous,
    # I define midpoint frame to mean midpoint of translation vector between cameras,
    # but still parallel with the left camera frame.
    t_OrelToR_inR = T / 2
    t_LrelTo0_inR = t_OrelToR_inR

    t_LrelToO_inL = R.T @ t_LrelTo0_inR

    print(f"t_LrelToO_inL: \n{t_LrelToO_inL}")

    t_Rrelto0_inL = -R.T @ t_OrelToR_inR
    print(f"t_Rrelto0_inL: \n{t_Rrelto0_inL}")

    ball_traj = ball_traj + t_LrelToO_inL.reshape(1, 3)

    #fit line to top view trajectory to get x coordinate @ z=0
    # fit parabola to side view to get y coordinate @ z=0
    best_fit_line = np.polyfit(ball_traj[:, 2], ball_traj[:, 0], 1)
    best_fit_parabola = np.polyfit(ball_traj[:, 2], ball_traj[:, 1], 2)

    #plot lines on top view and parabola on side view
    z = np.linspace(max(ball_traj[:, 2]), 0, 100)
    x = best_fit_line[0] * z + best_fit_line[1]
    y = best_fit_parabola[0] * z**2 + best_fit_parabola[
        1] * z + best_fit_parabola[2]

    time = np.arange(0, ball_traj.shape[0], 1)

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title('Side View')
    scatter = axs[0].scatter(ball_traj[:, 2],
                             ball_traj[:, 1],
                             c=time,
                             cmap='viridis')
    axs[0].scatter(t_LrelToO_inL[2],
                   t_LrelToO_inL[1],
                   c='r',
                   label='Left Camera',
                   marker='s')
    axs[0].scatter(t_Rrelto0_inL[2],
                   t_Rrelto0_inL[1],
                   c='b',
                   label='Right Camera',
                   marker='s')

    axs[0].scatter(0, 0, c='g', label='Midpoint', marker='x')
    axs[0].plot(z, y, 'r--', label='Best Fit Parabola')

    axs[0].set_xlim(max(ball_traj[:, 2]), min(ball_traj[:, 2]))
    axs[0].set_ylim(100, -100)
    axs[0].set_xlabel('Z')
    axs[0].set_ylabel('Y')
    axs[0].axis('equal')
    axs[0].grid()
    axs[0].legend()
    cbar0 = plt.colorbar(scatter)
    cbar0.set_label('Frame')

    axs[1].set_title('Top View')
    scatter = axs[1].scatter(ball_traj[:, 2],
                             ball_traj[:, 0],
                             c=time,
                             cmap='viridis')
    axs[1].scatter(t_LrelToO_inL[2],
                   t_LrelToO_inL[0],
                   c='r',
                   label='Left Camera',
                   marker='s')
    axs[1].scatter(t_Rrelto0_inL[2],
                   t_Rrelto0_inL[0],
                   c='b',
                   label='Right Camera',
                   marker='s')
    axs[1].plot(z, x, 'r--', label='Best Fit Line')
    axs[1].scatter(0, 0, c='g', label='Midpoint', marker='x')
    axs[1].set_xlim(max(ball_traj[:, 2]), min(ball_traj[:, 2]))
    axs[1].set_ylim(min(ball_traj[:, 0]), max(ball_traj[:, 0]))
    axs[1].set_xlabel('Z')
    axs[1].set_ylabel('X')
    axs[1].axis('equal')
    axs[1].grid()
    axs[1].legend()
    cbar1 = plt.colorbar(scatter)
    cbar1.set_label('Frame')

    print(f"R (rotation right to left): \n{R}")
    print(f"T (position of left relative to right, expressed in right): \n{T}")
    print(f"Estimated x intercept: {best_fit_line[1]}")
    print(f"Estimated y intercept: {best_fit_parabola[2]}")
    plt.show()
