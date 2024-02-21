import os
import cv2 as cv
import numpy as np
import glob


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

        self.blob_detector = cv.SimpleBlobDetector_create(params)

    def _get_place_to_look_for_ball(self, img):
        #remove background
        diff = cv.absdiff(img, self.background)

        #  gaussian blur diff image
        blur = cv.medianBlur(diff, 5)

        #use time history to only look in region of interest so that thresholds can be low.
        time_diff = cv.absdiff(blur, self.prev_image)

        #threshold blur on about gresyscale of 5ish
        ret, thresh = cv.threshold(blur, 7, 255, cv.THRESH_BINARY)

        # erosion to remove noise and fill in gaps
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv.dilate(thresh, kernel, iterations=3)
        eroded = cv.erode(dilated, kernel, iterations=3)

        #find contour with largest bounding box
        contours, hierarchy = cv.findContours(eroded, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)
        #get max area bounding box
        max_area_idx = 0
        print(len(contours))
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

            cv.rectangle(
                self.plot_img, (self.x_bound, self.y_bound),
                (self.x_bound + self.w_bound, self.y_bound + self.h_bound),
                (255, 0, 0), 2)

            #crop everything inside of bounding box
            crop_img = eroded[self.y_bound:self.y_bound + self.h_bound,
                              self.x_bound:self.x_bound + self.w_bound]
        else:
            crop_img = eroded

        self.prev_image = blur

        return diff, blur, thresh, eroded, crop_img, time_diff

    def _get_ball_circle(self, img):

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
                # cv.circle(self.plot_img, (x, y), circ_r, (0, 255, 0), 2)
                cv.rectangle(self.plot_img, (x - 2, y - 2), (x + 2, y + 2),
                             (0, 0, 255), -1)

        # keypoints = self.blob_detector.detect(img)
        # img_with_keypoints = cv.drawKeypoints(
        #     img, keypoints, np.array([]), (0, 0, 255),
        #     cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return None

    def detect(self, img, display=False):
        self.plot_img = img.copy()
        if not self.incoming_in_gray:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if not self.first_detect:
            self.background = img.copy()
            self.first_detect = True

        diff, blur, thresh, eroded, crop_img, time_diff = self._get_place_to_look_for_ball(
            img)
        circle = self._get_ball_circle(crop_img)

        if self.display:
            cv.imshow('raw', self.plot_img)
            cv.imshow('diff', diff)
            cv.imshow('blur', blur)
            cv.imshow('thresh', thresh)
            cv.imshow('dilated and eroded', eroded)
            cv.imshow('cropped', crop_img)
            cv.imshow('time diff', time_diff)
            # cv.imshow('background', self.background)
            # cv.imshow('circle', circle)
            key = cv.waitKey(0)

            if key == ord('q'):
                cv.destroyAllWindows()
                raise SystemExit
            elif key == ord('s'):
                cv.imwrite('cropped.png', crop_img)

        # raise NotImplementedError


if __name__ == '__main__':

    baseball_detector = BaseballDetector(grayscale=False)

    left_images = sorted(glob.glob('./20240215112959/L/*.png'),
                         key=lambda filename: int(
                             os.path.splitext(os.path.basename(filename))[0]))

    for i, filepath in enumerate(left_images):
        img = cv.imread(filepath)
        baseball_detector.detect(img, display=True)

    # #read in one image at a time, find the ball, and display the image with a rectangle overlay
    # # detector = cv.SimpleBlobDetector()
    # prev_img = np.zeros(images[0][0].shape, dtype=np.uint8)

    # #make video writer for mp4
    # out = cv.VideoWriter('baseball_tracking.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 10, (640, 480))

    # for i in range(len(images)):
    #     first_time = True

    #     for i, img in enumerate(images[i]):
    #         plot_img = img.copy()
    #         # find absdiff
    #         diff = cv.absdiff(img, prev_img)
    #         diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    #         # gaussian blur diff image
    #         blur = cv.GaussianBlur(diff_gray, (3, 3), 0)

    #         # erosion to remove noise
    #         kernel = np.ones((5, 5), np.uint8)
    #         diff_gray = cv.dilate(blur, kernel, iterations=4)
    #         diff_gray = cv.erode(diff_gray, kernel, iterations=4)

    #         #threshold blur on about gresyscale of 5ish
    #         ret, thresh = cv.threshold(diff_gray, 10, 255, cv.THRESH_BINARY)

    #         #find contour with largest bounding box
    #         contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #         #get max area bounding box
    #         max_area_idx = 0
    #         for i, contour in enumerate(contours):
    #             x, y, w, h = cv.boundingRect(contour)
    #             area = cv.contourArea(contour)
    #             if area > cv.contourArea(contours[max_area_idx]):
    #                 max_area_idx = i
    #         #draw bounding box
    #         x, y, w, h = cv.boundingRect(contours[max_area_idx])
    #         if not first_time:
    #             pass
    #             # cv.rectangle(plot_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    #         #zero out everything outside of bounding box
    #         mask = np.zeros(thresh.shape, np.uint8)
    #         mask[y:y + h, x:x + w] = 255
    #         thresh = cv.bitwise_and(img, img, mask=mask)

    #         blur2 = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

    #         # #round of dilation and kernel, iterations=1)
    #         #blur image a bit per docs suggestion
    #         # blur2 = cv.GaussianBlur(thresh, (15, 15), 0)

    #         if not first_time:
    #             #find circles
    #             circles = cv.HoughCircles(blur2,
    #                                     cv.HOUGH_GRADIENT,
    #                                     5,
    #                                     100,
    #                                     param1=50,
    #                                     param2=10,
    #                                     minRadius=5,
    #                                     maxRadius=25)

    #             if circles is not None:
    #                 circles = np.round(circles[0, :]).astype("int")
    #                 for (circ_x, circ_y, circ_r) in circles:
    #                     #onyl plot circles that are within the bounding box

    #                     if circ_x < (x + w) and circ_x > x and circ_y < (y + h) and circ_y > y:
    #                         # pass
    #                         cv.circle(plot_img, (circ_x, circ_y), circ_r, (0, 255, 0), 2)
    #                         cv.rectangle(plot_img, (circ_x - 5, circ_y - 5), (circ_x + 5, circ_y + 5), (0, 128, 255), -1)

    #         # keypoints = detector.detect(img)
    #         # img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #         cv.imshow('diff Image', diff_gray)
    #         # cv.imshow('first thresh', first_thresh)
    #         cv.imshow('Current Image', plot_img)
    #         cv.imshow('thresh', thresh)
    #         cv.imshow('blur', blur2)
    #         first_time = False
    #         #kill program on 'q' key press or proceed

    #         #make video of images
    #         out.write(plot_img)

    #         key = cv.waitKey(0)
    #         prev_img = img
    #         if key == ord('q'):
    #             break
