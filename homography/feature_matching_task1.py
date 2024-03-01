import numpy as np
import cv2 as cv

# https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

cropped_img = cv.imread('old_cropped.jpg')
full_img = cv.imread('old_full.jpg')


def extract_features(image):
    #create feature extractor and extract features from both images
    orb = cv.SIFT_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def match_features(cropped_des, full_des):
    #create a brute force matcher and match the features
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(cropped_des, full_des)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches



cropped_kp, cropped_des = extract_features(cropped_img)
#? is there a way to use the bounding box as a mask to look for features?
full_kp, full_des = extract_features(full_img)

#bounding box for the picture is top left, top right, bottom left, and bottom right of cropped image

matches = match_features(cropped_des.astype(np.uint8),
                         full_des.astype(np.uint8))

# Find homography
best_matches = matches[:20]
src_pts = np.float32([cropped_kp[m.queryIdx].pt
                      for m in best_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([full_kp[m.trainIdx].pt
                      for m in best_matches]).reshape(-1, 1, 2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# Define bounding box points
pts = np.float32([[2, 0], [1, 149], [191, 151], [195,2]]).reshape(-1, 1, 2)

# Transform bounding box
dst = cv.perspectiveTransform(pts, M)

# Draw bounding box
full_img_bounding_box = cv.polylines(full_img, [np.int32(dst)], True, 255, 2,
                                     cv.LINE_AA)

img3 = cv.drawMatches(cropped_img,
                      cropped_kp,
                      full_img,
                      full_kp,
                      matches[:20],
                      None,
                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Matches', img3)
# cv.waitKey(0)
# cv.imshow('Matches & Bounding Box', full_img_bounding_box)
cv.waitKey(0)
