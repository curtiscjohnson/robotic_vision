import cv2 as cv
import numpy as np

img = cv.imread('./cropped.png')

# params = cv.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 0
# params.maxThreshold = 0

# # Filter by Area
# params.filterByArea = False
# params.minArea = 1500

# # Filter by Circularity
# params.filterByCircularity = False
# params.minCircularity = 0.1

# # Filter by Convexity
# params.filterByConvexity = False
# params.minConvexity = 0.1

# # Filter by Inertia
# params.filterByInertia = False
# params.minInertiaRatio = 0.01

# # Create a detector with the parameters
# detector = cv.SimpleBlobDetector_create(params)

# keypoints = detector.detect(img)
# img_with_keypoints = cv.drawKeypoints(
#     img, keypoints, np.array([]), (0, 0, 255),
#     cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv.imshow('keypoints', img_with_keypoints)
# cv.waitKey(0)

#find circles
circles = cv.HoughCircles(img,
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
        x = circ_x
        y = circ_y
        #onyl plot circles that are within the bounding box
        cv.circle(img, (x, y), circ_r, (0, 255, 0), 2)
        cv.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
