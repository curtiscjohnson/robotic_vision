import zipfile
import os
import cv2 as cv
import numpy as np
import time

def extract_images(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    # Load images into a list
    left_images = []
    right_images = []
    filepaths = []
    for file in os.listdir('./Sequence1/'):
        if file.endswith(".jpg") or file.endswith(".png"):
            filepaths.append(os.path.join('./Sequence1/', file))

    #sort the filepaths so that the images are in order
    filepaths.sort()

    for file in filepaths:
        img = cv.imread(file)
        if "L" in file:
            left_images.append(img)
        elif "R" in file:
            right_images.append(img)

    return left_images, right_images


# specify the zip file path and the directory to extract to
zip_path = 'Baseball Practice Images.zip'
images = extract_images(zip_path)
print(f"Extracted {len(images[0]) + len(images[1])} images")

#read in one image at a time, find the ball, and display the image with a rectangle overlay
# detector = cv.SimpleBlobDetector()
prev_img = np.zeros(images[0][0].shape, dtype=np.uint8)

#make video writer for mp4
out = cv.VideoWriter('baseball_tracking.mp4', cv.VideoWriter_fourcc('m','p','4','v'), 10, (640, 480))

for i in range(len(images)):
    first_time = True

    for i, img in enumerate(images[i]):
        plot_img = img.copy()
        # find absdiff
        diff = cv.absdiff(img, prev_img)
        diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

        # gaussian blur diff image
        blur = cv.GaussianBlur(diff_gray, (3, 3), 0)

        # erosion to remove noise
        kernel = np.ones((5, 5), np.uint8)
        diff_gray = cv.dilate(blur, kernel, iterations=4)
        diff_gray = cv.erode(diff_gray, kernel, iterations=4)

        #threshold blur on about gresyscale of 5ish
        ret, thresh = cv.threshold(diff_gray, 10, 255, cv.THRESH_BINARY)

        #find contour with largest bounding box
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #get max area bounding box
        max_area_idx = 0
        for i, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            area = cv.contourArea(contour)
            if area > cv.contourArea(contours[max_area_idx]):
                max_area_idx = i
        #draw bounding box
        x, y, w, h = cv.boundingRect(contours[max_area_idx])
        if not first_time:
            pass
            # cv.rectangle(plot_img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        #zero out everything outside of bounding box
        mask = np.zeros(thresh.shape, np.uint8)
        mask[y:y + h, x:x + w] = 255
        thresh = cv.bitwise_and(img, img, mask=mask)
        
        blur2 = cv.cvtColor(thresh, cv.COLOR_BGR2GRAY)

        # #round of dilation and kernel, iterations=1)
        #blur image a bit per docs suggestion
        # blur2 = cv.GaussianBlur(thresh, (15, 15), 0)

        if not first_time:
            #find circles
            circles = cv.HoughCircles(blur2,
                                    cv.HOUGH_GRADIENT,
                                    5,
                                    100,
                                    param1=50,
                                    param2=10,
                                    minRadius=5,
                                    maxRadius=25)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (circ_x, circ_y, circ_r) in circles:
                    #onyl plot circles that are within the bounding box
                    
                    if circ_x < (x + w) and circ_x > x and circ_y < (y + h) and circ_y > y:
                        # pass
                        cv.circle(plot_img, (circ_x, circ_y), circ_r, (0, 255, 0), 2)
                        cv.rectangle(plot_img, (circ_x - 5, circ_y - 5), (circ_x + 5, circ_y + 5), (0, 128, 255), -1)


        # keypoints = detector.detect(img)
        # img_with_keypoints = cv.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('diff Image', diff_gray)
        # cv.imshow('first thresh', first_thresh)
        cv.imshow('Current Image', plot_img)
        cv.imshow('thresh', thresh)
        cv.imshow('blur', blur2)
        first_time = False
        #kill program on 'q' key press or proceed


        #make video of images
        out.write(plot_img)

        key = cv.waitKey(0)
        prev_img = img
        if key == ord('q'):
            break