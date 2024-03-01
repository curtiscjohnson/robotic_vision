import cv2 as cv
import numpy as np
import time


def extract_features(image):
    #create feature extractor and extract features from both images
    kp, des = orb.detectAndCompute(image, None)
    return kp, des


def match_features(cropped_des, full_des):
    # FLANN parameters
    matches = flann.knnMatch(cropped_des, full_des, k=2)
    good_matches = []
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1, 0]
            good_matches.append([m])

    #return best 20 matches
    # good_matches = sorted(good_matches, key=lambda x: x[0].distance)

    return good_matches, matchesMask


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

orb = cv.SIFT_create()

target = cv.imread("./new.jpeg", cv.IMREAD_GRAYSCALE)
reference = cv.imread("./old_cropped.jpg", cv.IMREAD_GRAYSCALE)
reference = cv.resize(reference,
                      (reference.shape[1] // 4, reference.shape[0] // 4),
                      reference, 0, 0, cv.INTER_AREA)

target = cv.resize(target, (reference.shape[1], reference.shape[0]), target, 0,
                   0, cv.INTER_AREA)

cv.convertScaleAbs(target, target, 0.9, -50)

ref_kp, ref_des = extract_features(reference)

# Create a VideoCapture object
cap = cv.VideoCapture('video.mp4')

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4),
                          frame, 0, 0, cv.INTER_AREA)

        start = time.time()
        video_kp, video_des = extract_features(frame)
        print("Time to extract features: ", time.time() - start)

        start = time.time()
        matches, mask = match_features(ref_des.astype(np.float32),
                                       video_des.astype(np.float32))
        print("Time to match features: ", time.time() - start)

        start = time.time()
        img3 = cv.drawMatchesKnn(
            reference,
            ref_kp,
            frame,
            video_kp,
            matches,
            None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        print("Time to draw matches: ", time.time() - start)

        # # Find homography
        # best_matches = matches[:20]
        # src_pts = np.float32([ref_kp[m.queryIdx].pt
        #                     for m in best_matches]).reshape(-1, 1, 2)
        # dst_pts = np.float32([video_kp[m.trainIdx].pt
        #                     for m in best_matches]).reshape(-1, 1, 2)

        # M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # #map target image to video image
        # h, w, _ = reference.shape
        # img_warped = cv.warpPerspective(target, M, (frame.shape[1], frame.shape[0]))

        # Display the resulting frame
        cv.imshow('Frame', img3)

        # Press Q on keyboard to exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
