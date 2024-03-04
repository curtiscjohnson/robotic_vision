import cv2 as cv
import numpy as np


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
    print(y)
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
    print(x)
    min_x = x[0]

    num_circles_on_left = np.argmax(x > min_x + threshold)
    print(f"num_circles_on_left: {num_circles_on_left}")

    #right will have largest x values
    max_x = x[-1]
    num_circles_on_right = len(x) - np.argmax(x > max_x - threshold)
    print(f"num_circles_on_right: {num_circles_on_right}")


def main():
    ref = cv.imread('./images/Blue.jpg')
    ref = cv.resize(ref, (0, 0), fx=0.25, fy=0.25)

    blue = cv.imread('./images/Blue.jpg', cv.IMREAD_GRAYSCALE)
    blue = cv.resize(blue, ref.shape[:2][::-1])
    blue = cv.medianBlur(blue, 5)
    blue = cv.threshold(blue, 50, 255, cv.THRESH_BINARY_INV)[1]

    circles = find_circles(blue)
    count_circles_on_sides(circles)
    ref = draw_circles(ref, circles)

    # cv.imshow('Blue Circles', blue)
    cv.imshow('Reference', ref)
    cv.waitKey(0)


if __name__ == '__main__':
    main()

    cv.destroyAllWindows()
