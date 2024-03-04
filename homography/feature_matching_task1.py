import numpy as np
import cv2 as cv

# https://stackoverflow.com/questions/51606215/how-to-draw-bounding-box-on-best-matches
# https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

# cropped_img = cv.imread('old_cropped.jpg')
# full_img = cv.imread('old_full.jpg')

# cropped_img = cv.imread('./robotics_book_cropped.png')
# cropped_img = cv.GaussianBlur(cropped_img, (5, 5), 3)
# full_img = cv.imread('./roboticsbookfull.jpg')
# full_img = cv.GaussianBlur(full_img, (5, 5), 3)

# cropped_img = cv.imread('./nonlinear_cropped.png')
# full_img = cv.imread('./nonlinear_full.jpg')

cropped_img = cv.imread('./poster.jpg')
cropped_img = cv.resize(cropped_img,
                        (cropped_img.shape[1] // 4, cropped_img.shape[0] // 4),
                        cropped_img, 0, 0, cv.INTER_AREA)
full_img = cv.imread('./poster_full.jpg')
full_img = cv.resize(full_img,
                     (full_img.shape[1] // 1, full_img.shape[0] // 1),
                     full_img, 0, 0, cv.INTER_AREA)

# cropped_img = cv.imread("./lab_sign.jpg")
# cropped_img = cv.GaussianBlur(cropped_img, (5, 5), 3)
# full_img = cv.imread("./lab_sign_full.jpg")
# full_img = cv.GaussianBlur(full_img, (5, 5), 3)


def extract_features(image):
    #create feature extractor and extract features from both images
    orb = cv.SIFT_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def match_features(cropped_des, full_des):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=2)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    # FLANN parameters
    matches = flann.knnMatch(cropped_des.astype(np.float32),
                             full_des.astype(np.float32),
                             k=2)

    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.5 * m2.distance]

    #return best 20 matches
    good = sorted(good, key=lambda x: x.distance)

    return good


cropped_kp, cropped_des = extract_features(cropped_img)
#? is there a way to use the bounding box as a mask to look for features?
full_kp, full_des = extract_features(full_img)

#bounding box for the picture is top left, top right, bottom left, and bottom right of cropped image

matches = match_features(cropped_des.astype(np.uint8),
                         full_des.astype(np.uint8))

# Find homography
numToUse = 500
print(len(matches))
best_matches = matches[:numToUse]
src_pts = np.float32([cropped_kp[m.queryIdx].pt
                      for m in best_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([full_kp[m.trainIdx].pt
                      for m in best_matches]).reshape(-1, 1, 2)

M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# Define bounding box points
w, h = cropped_img.shape[1], cropped_img.shape[0]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1,
                                                       0]]).reshape(-1, 1, 2)
# pts = np.float32([[0, 0], [1, 149], [191, 151], [195, 2]]).reshape(-1, 1, 2)

# Transform bounding box
dst = cv.perspectiveTransform(pts, M)

# Draw bounding box
full_img_bounding_box = cv.polylines(full_img, [np.int32(dst)], True, 255, 2,
                                     cv.LINE_AA)

img3 = cv.drawMatches(cropped_img,
                      cropped_kp,
                      full_img,
                      full_kp,
                      matches[:numToUse],
                      None,
                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv.imshow('Matches', img3)
# cv.waitKey(0)
# cv.imshow('Matches & Bounding Box', full_img_bounding_box)
cv.waitKey(0)
