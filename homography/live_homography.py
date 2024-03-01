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

    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    #return best 20 matches
    # good_matches = sorted(good_matches, key=lambda x: x[0].distance)

    return good


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
orb = cv.SIFT_create(nfeatures=0,
                     nOctaveLayers=3,
                     contrastThreshold=0.0001,
                     edgeThreshold=100,
                     sigma=1.6)

target = cv.imread("./new.jpeg")
reference = cv.imread("./old_cropped.jpg")
reference = cv.resize(reference,
                      (reference.shape[1] // 1, reference.shape[0] // 1),
                      reference, 0, 0, cv.INTER_AREA)

target = cv.resize(target, (reference.shape[1], reference.shape[0]), target, 0,
                   0, cv.INTER_AREA)


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
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3),
                          frame, 0, 0, cv.INTER_AREA)

        start = time.time()
        video_kp, video_des = extract_features(frame)
        print("Time to extract features: ", time.time() - start)

        start = time.time()
        matches = match_features(ref_des.astype(np.float32),
                                 video_des.astype(np.float32))
        print("Time to match features: ", time.time() - start)

        start = time.time()
        img3 = cv.drawMatches(reference,
                              ref_kp,
                              frame,
                              video_kp,
                              matches,
                              None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        print("Time to draw matches: ", time.time() - start)

        # Find homography
        # best_matches = matches[:20]
        start = time.time()
        src_pts = np.float32([ref_kp[m.queryIdx].pt
                              for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([video_kp[m.trainIdx].pt
                              for m in matches]).reshape(-1, 1, 2)

        #check if there are at least 4 points
        if len(src_pts) < 4 or len(dst_pts) < 4:
            print(len(src_pts), len(dst_pts))
            raise Exception("Not enough points to find homography")

        print("Using ", len(src_pts), " points to find homography.")
        M, mask = cv.findHomography(src_pts,
                                    dst_pts,
                                    cv.RANSAC,
                                    9.0,
                                    maxIters=1000,
                                    confidence=0.999)

        #map target image to video image
        h, w, _ = reference.shape
        img_warped = cv.warpPerspective(target, M,
                                        (frame.shape[1], frame.shape[0]))
        print("Time to warp image: ", time.time() - start)

        # cv.imshow('Warped Image', img_warped)

        #create mask based on warped image
        img_warped_gray = cv.cvtColor(img_warped, cv.COLOR_BGR2GRAY)
        # cv.imshow('Warped Image Gray', img_warped_gray)
        _, mask = cv.threshold(img_warped_gray, 0, 255, cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        # cv.imshow('Mask', mask_inv)

        #use mask to remove reference image from video image
        img1_bg = cv.bitwise_and(frame, frame, mask=mask_inv)

        #put target image on top of video image
        img3 = cv.add(img1_bg, img_warped)
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
