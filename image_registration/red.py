import os
import cv2 as cv
import numpy as np
import glob

np.set_printoptions(precision=2, suppress=True, linewidth=200)


def find_circles(image):
    circles = cv.HoughCircles(image,
                              cv.HOUGH_GRADIENT,
                              1,
                              50,
                              param1=50,
                              param2=10,
                              minRadius=5,
                              maxRadius=10)

    return np.array(circles)


def draw_circles(image, circles):
    if circles is not None:
        circles = circles[0, :, :]
        circles = sorted(circles, key=lambda x: x[2])
        for i in circles:
            i = i.astype(int)

            cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image


def count_circles_on_sides(circles):
    threshold = 5
    #top will have smallest y values, sort by y values, then see when y values changes too much
    circles_y = circles[0, :, 1]
    y = np.sort(circles_y)
    # print(y)
    min_y = y[0]

    #find first index where y value changes too much
    num_circles_on_top = np.argmax(y > min_y + threshold)
    print(f"num_circles_on_top: {num_circles_on_top}")

    #bottom will have largest y values
    max_y = y[-1]
    num_circles_on_bottom = len(y) - np.argmax(y > max_y - threshold)
    print(f"num_circles_on_bottom: {num_circles_on_bottom}")

    #left will have smallest x values
    circles_x = circles[0, :, 0]
    x = np.sort(circles_x)
    # print(x)
    min_x = x[0]

    num_circles_on_left = np.argmax(x > min_x + threshold)
    print(f"num_circles_on_left: {num_circles_on_left}")

    #right will have largest x values
    max_x = x[-1]
    num_circles_on_right = len(x) - np.argmax(x > max_x - threshold)
    print(f"num_circles_on_right: {num_circles_on_right}")

    return num_circles_on_top, num_circles_on_bottom, num_circles_on_left, num_circles_on_right


def rotate_to_reference(image, numCirclesOnSides):
    six_side = np.argmax(numCirclesOnSides)

    if six_side == 0:
        return image
    elif six_side == 1:
        return cv.rotate(image, cv.ROTATE_180)
    elif six_side == 2:
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif six_side == 3:
        return cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)


def grid_image(image):
    plot_image = image.copy()
    plot_image = cv.cvtColor(plot_image, cv.COLOR_GRAY2BGR)
    h, w = image.shape[:2]

    #grid up answers with linspace in answer field, should organize by question number and answer
    x_grid = np.linspace(13, 140, 11, dtype=np.uint16)
    y_grid = np.linspace(7, 632, 26, dtype=np.uint16)

    #draw grid on imageq
    for x in x_grid:
        cv.line(plot_image, (x, 0), (x, h), (0, 255, 0), 1)

    for y in y_grid:
        cv.line(plot_image, (0, y), (w, y), (255, 0, 0), 1)

    #img[x:x+1, y:y+1] is a single answer field
    #for each rectangle in grid, get grid number with least amount of white pixels

    return plot_image, x_grid, y_grid


def sum_pixels_in_grid(image, xgrid, ygrid):

    grid_sums = np.zeros((len(xgrid) - 1, len(ygrid) - 1), dtype=np.float32)
    #get sum of pixels in each grid
    for i in range(len(xgrid) - 1):
        for j in range(len(ygrid) - 1):
            max_sum = 255 * image[ygrid[j]:ygrid[j + 1],
                                  xgrid[i]:xgrid[i + 1]].size
            # print(
            #     f"Searching in grid: {xgrid[i],xgrid[i+1]}, {ygrid[j],ygrid[j+1]}"
            # )

            sum = np.sum(image[ygrid[j]:ygrid[j + 1], xgrid[i]:xgrid[i + 1]],
                         dtype=np.uint32)

            # print(f"Sum of pixels in grid: {sum}")

            grid_sums[i, j] = sum / max_sum

            # sum = np.sum(image[ygrid[y]:ygrid[y + 1], xgrid[x]:xgrid[x + 1]],
            #              dtype=np.uint32)
            # print(f"Sum of pixels in grid: {sum}")

            #write text on with sum of pixels in each grid
            cv.putText(image, f"{np.round(sum/max_sum, 2)}",
                       (xgrid[i] + 3, ygrid[j] + 10), cv.FONT_HERSHEY_SIMPLEX,
                       0.25, (0, 0, 255), 1, cv.LINE_AA)

    # cv.imshow('Grid Sums', image)
    # cv.waitKey(0)

    # print(grid_sums / np.max(grid_sums))
    # print(grid_sums.shape)

    return grid_sums


def extract_answers(grid_sums):
    letter_answer = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    answer_idx = np.argmax(grid_sums, axis=0)
    print(np.min(np.max(grid_sums, axis=0)))

    return [letter_answer[i] for i in answer_idx]


def crop_only_answers(image):
    answers = image[168:808, 25:618]
    plot = answers.copy()
    plot = cv.cvtColor(plot, cv.COLOR_GRAY2BGR)

    h, w = answers.shape[:2]
    answerbox1 = answers[0:h, 0:147]
    answerbox2 = answers[0:h, 147:296]
    answerbox3 = answers[0:h, 296:446]
    answerbox4 = answers[0:h, 446:w]

    # #draw verical line on image
    # cv.line(plot, (296, 0), (296, h), (0, 255, 0), 1)
    # cv.imshow('Answers', plot)

    #make all images same size
    return answerbox1, answerbox2, answerbox3, answerbox4


def main():
    ref = cv.imread('./images/Red.jpg')
    ref = cv.resize(ref, (0, 0), fx=0.25, fy=0.25)
    ref_plot = ref.copy()
    # ref_features = ref[:170, :]
    ref_features = ref

    ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)
    # ref = cv.threshold(ref, 200, 255, cv.THRESH_BINARY)[1]
    ref = cv.adaptiveThreshold(ref, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 51, 50)
    # ref = cv.GaussianBlur(ref, (5, 5), 0)
    ref = cv.medianBlur(ref, 7)

    # cv.imshow('Reference', ref)
    red_images = glob.glob('./images/Red *.jpg')

    orb = cv.SIFT_create()

    ref_kp, ref_des = orb.detectAndCompute(ref_features, None)
    ref_plot = cv.drawKeypoints(ref, ref_kp, ref_plot)

    # cv.imshow('Corners', ref_plot)

    for red_file in red_images:
        print(red_file)
        red = cv.imread(red_file)
        red = cv.resize(red, (0, 0), fx=0.25, fy=0.25)
        red_plot = red.copy()
        #set red channel to 0
        red = cv.cvtColor(red, cv.COLOR_BGR2GRAY)
        red_features = red
        red = cv.adaptiveThreshold(red, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 51, 50)

        # red = cv.GaussianBlur(red, (5, 5), 0)
        red = cv.medianBlur(red, 7)
        # red_features = red[:170, :]
        cv.imshow('red', red)

        red_kp, blue_des = orb.detectAndCompute(red_features, None)
        red_plot = cv.drawKeypoints(red, red_kp, red_plot)

        #match features between ref_kp and red_kp
        bf = cv.BFMatcher()
        matches = bf.knnMatch(ref_des, blue_des, k=2)
        good = [m1 for m1, m2 in matches if m1.distance < 0.45 * m2.distance]

        # Assuming good is your list of matches
        ref_pts = np.float32([ref_kp[m.queryIdx].pt
                              for m in good]).reshape(-1, 1, 2)
        red_pts = np.float32([red_kp[m.trainIdx].pt
                              for m in good]).reshape(-1, 1, 2)

        red_plot = cv.drawMatchesKnn(ref_features,
                                     ref_kp,
                                     red_features,
                                     red_kp, [good],
                                     red_plot,
                                     flags=2)

        # cv.imshow('Feature Matching', red_plot)

        H, mask = cv.findHomography(red_pts, ref_pts, cv.RANSAC, 5.0)

        # #map red image to reference image
        red = cv.warpPerspective(red, H, ref.shape[:2][::-1])

        # cv.imshow('Blue after warp', red)

        diff = cv.absdiff(ref, red)

        cv.imshow('Diff', diff)

        #detect vertical lines in diff image
        answer_field1, answer_field2, answer_field3, answer_field4 = crop_only_answers(
            diff)

        # cv.imshow('Answer Field1', answer_field1)
        # cv.imshow('Answer Field2', answer_field2)
        # cv.imshow('Answer Field3', answer_field3)
        # cv.imshow('Answer Field4', answer_field4)

        grid1, _, _ = grid_image(answer_field1)
        grid2, _, _ = grid_image(answer_field2)
        grid3, _, _ = grid_image(answer_field3)
        grid4, xgrid, ygrid = grid_image(answer_field4)

        cv.imshow('Grid1', grid1)
        cv.imshow('Grid2', grid2)
        cv.imshow('Grid3', grid3)
        cv.imshow('Grid4', grid4)

        grid1_sums = sum_pixels_in_grid(grid1, xgrid, ygrid)
        grid2_sums = sum_pixels_in_grid(grid2, xgrid, ygrid)
        grid3_sums = sum_pixels_in_grid(grid3, xgrid, ygrid)
        grid4_sums = sum_pixels_in_grid(grid4, xgrid, ygrid)

        grid1_letters = extract_answers(grid1_sums)
        grid2_letters = extract_answers(grid2_sums)
        grid3_letters = extract_answers(grid3_sums)
        grid4_letters = extract_answers(grid4_sums)

        # print(grid1_sums)
        # print(grid1_letters)
        # print()
        # print(grid2_sums)
        # print(grid2_letters)
        # print()
        # print(grid3_sums)
        # print(grid3_letters)
        # print()
        # print(grid4_sums)
        # print(grid4_letters)
        # print()

        letters = grid1_letters + grid2_letters + grid3_letters + grid4_letters

        print(red_file.split(".jpg"))
        filename = os.path.basename(red_file).split(".jpg")[0]
        color = filename.split(" ")[0]
        number = filename.split(" ")[1]

        if not os.path.exists('./answers'):
            os.makedirs('./answers')

        np.savetxt(f'./answers/{color} Output {number}.txt', letters, fmt='%s')

        # key = cv.waitKey(0)
        # if key == ord('q'):
        #     break


if __name__ == '__main__':
    main()

    cv.destroyAllWindows()
