import cv2 as cv


def find_circles(image):
    circles = cv.HoughCircles(image,
                              cv.HOUGH_GRADIENT,
                              1,
                              50,
                              param1=50,
                              param2=10,
                              minRadius=5,
                              maxRadius=10)

    return circles


def draw_circles(image, circles):
    if circles is not None:
        circles = circles[0, :, :]
        circles = sorted(circles, key=lambda x: x[2])
        for i in circles:
            i = i.astype(int)

            cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image


def main():
    ref = cv.imread('./images/Blue.jpg')
    ref = cv.resize(ref, (0, 0), fx=0.25, fy=0.25)

    blue = cv.imread('./images/Blue.jpg', cv.IMREAD_GRAYSCALE)
    blue = cv.resize(blue, ref.shape[:2][::-1])
    blue = cv.medianBlur(blue, 5)
    # blue = cv.GaussianBlur(blue, (5, 5), 3)
    # cv.imshow('Before threshold', blue)
    # blue = cv.adaptiveThreshold(blue, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    blue = cv.threshold(blue, 50, 255, cv.THRESH_BINARY_INV)[1]

    circles = find_circles(blue)
    ref = draw_circles(ref, circles)

    cv.imshow('Blue Circles', blue)
    cv.imshow('Reference', ref)
    cv.waitKey(0)


if __name__ == '__main__':
    main()

    cv.destroyAllWindows()
