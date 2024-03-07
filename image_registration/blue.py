import os
import cv2 as cv
import numpy as np
import glob

np.set_printoptions(precision=2, suppress=True)


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
    x_grid = np.linspace(5, 486, 21, dtype=np.uint16)
    y_grid = np.linspace(22, 144, 11, dtype=np.uint16)

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
    answer_idx = np.argmax(grid_sums, axis=1)

    #reverse order of answer index to go up by question number
    answer_idx = answer_idx[::-1]

    return [letter_answer[i] for i in answer_idx]


def crop_only_answers(image):
    answers = image[332:786, 38:530]

    h, w = answers.shape[:2]

    answerbox1 = answers[0:h // 3, 0:w]
    answerbox2 = answers[h // 3:2 * h // 3, 0:w]
    answerbox3 = answers[2 * h // 3:h, 0:w]

    #make all images same size
    return answerbox1, answerbox2, answerbox3


def main():
    ref = cv.imread('./images/Blue.jpg')
    ref = cv.resize(ref, (0, 0), fx=0.25, fy=0.25)
    ref_plot = ref.copy()

    ref = cv.cvtColor(ref, cv.COLOR_BGR2GRAY)

    cv.imshow('Reference', ref)
    blue_images = glob.glob('./images/Blue *.jpg')

    orb = cv.SIFT_create()

    ref_kp, ref_des = orb.detectAndCompute(ref, None)
    ref_plot = cv.drawKeypoints(ref, ref_kp, ref_plot)

    # cv.imshow('Corners', ref_plot)

    for blue_file in blue_images:
        print(blue_file)
        blue = cv.imread(blue_file)
        blue = cv.resize(blue, (0, 0), fx=0.25, fy=0.25)
        cv.imshow('Blue', blue)
        blue_plot = blue.copy()
        blue = cv.cvtColor(blue, cv.COLOR_BGR2GRAY)

        blue_kp, blue_des = orb.detectAndCompute(blue, None)
        blue_plot = cv.drawKeypoints(blue, blue_kp, blue_plot)

        #match features between ref_kp and blue_kp
        bf = cv.BFMatcher()
        matches = bf.knnMatch(ref_des, blue_des, k=2)
        good = [m1 for m1, m2 in matches if m1.distance < 0.5 * m2.distance]

        # Assuming good is your list of matches
        ref_pts = np.float32([ref_kp[m.queryIdx].pt
                              for m in good]).reshape(-1, 1, 2)
        blue_pts = np.float32([blue_kp[m.trainIdx].pt
                               for m in good]).reshape(-1, 1, 2)

        blue_plot = cv.drawMatchesKnn(ref,
                                      ref_kp,
                                      blue,
                                      blue_kp, [good],
                                      blue_plot,
                                      flags=2)

        cv.imshow('Feature Matching', blue_plot)

        H, mask = cv.findHomography(blue_pts, ref_pts, cv.RANSAC, 5.0)

        #map blue image to reference image
        blue = cv.warpPerspective(blue, H, ref.shape[:2][::-1])

        # cv.imshow('Blue after warp', blue)

        diff = cv.absdiff(ref, blue)

        # cv.imshow('Diff', diff)

        answer_field1, answer_field2, answer_field3 = crop_only_answers(diff)

        cv.imshow('Answer Field1', answer_field1)
        cv.imshow('Answer Field2', answer_field2)
        cv.imshow('Answer Field3', answer_field3)

        grid1, _, _ = grid_image(answer_field1)
        grid2, _, _ = grid_image(answer_field2)
        grid3, xgrid, ygrid = grid_image(answer_field3)

        cv.imshow('Grid2', grid2)
        cv.imshow('Grid3', grid3)

        grid1_sums = sum_pixels_in_grid(grid1, xgrid, ygrid)
        grid2_sums = sum_pixels_in_grid(grid2, xgrid, ygrid)
        grid3_sums = sum_pixels_in_grid(grid3, xgrid, ygrid)

        grid1_letters = extract_answers(grid1_sums)
        grid2_letters = extract_answers(grid2_sums)
        grid3_letters = extract_answers(grid3_sums)

        letters = grid1_letters + grid2_letters + grid3_letters

        print(letters)

        print(blue_file.split(".jpg"))
        filename = os.path.basename(blue_file).split(".jpg")[0]
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
