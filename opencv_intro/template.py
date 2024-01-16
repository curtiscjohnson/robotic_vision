from http.client import UNAUTHORIZED
import threading
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
from tkinter import Tk, Frame, Button, BOTH, Label, Scale, Radiobutton  # Graphical User Inetrface Stuff
from tkinter import font as tkFont
import tkinter as tk

from sympy import is_convex

camera = cv.VideoCapture(0)
width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
videoout = cv.VideoWriter('./Video.avi', cv.VideoWriter_fourcc(*'XVID'), 25,
                          (width, height))  # Video format

# Button Definitions
ORIGINAL = 0
BINARY = 1
EDGE = 2
LINE = 3
ABSDIFF = 4
RGB = 5
HSV = 6
CORNERS = 7
CONTOURS = 8


def cvMat2tkImg(arr):  # Convert OpenCV image Mat to image for display
    rgb = cv.cvtColor(arr, cv.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    return ImageTk.PhotoImage(img)


class App(Frame):
    def __init__(self, winname='OpenCV'):  # GUI Design

        self.root = Tk()
        self.stopflag = True
        self.buffer = np.ones((height, width, 3), dtype=np.uint8) * 255

        global helv18
        helv18 = tkFont.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - width / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - height / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        Frame.__init__(self, self.root)
        self.pack(fill=BOTH, expand=1)
        # capture and display the first frame
        ret0, frame = camera.read()
        image = cvMat2tkImg(frame)
        self.panel = Label(image=image)
        self.panel.image = image
        self.panel.pack(side="top")
        # buttons
        global btnQuit
        btnQuit = Button(text="Quit", command=self.quit)
        btnQuit['font'] = helv18
        btnQuit.pack(side='right', pady=2)
        global btnStart
        btnStart = Button(text="Start", command=self.startstop)
        btnStart['font'] = helv18
        btnStart.pack(side='right', pady=2)
        # sliders
        global Slider1, Slider2
        Slider2 = Scale(self.root,
                        from_=0,
                        to=255,
                        length=255,
                        orient='horizontal')
        Slider2.pack(side='right')
        Slider2.set(192)
        Slider1 = Scale(self.root,
                        from_=0,
                        to=255,
                        length=255,
                        orient='horizontal')
        Slider1.pack(side='right')
        Slider1.set(64)
        # radio buttons
        global mode
        mode = tk.IntVar()
        mode.set(ORIGINAL)
        Radiobutton(self.root, text="Original", variable=mode,
                    value=ORIGINAL).pack(side='left', pady=4)
        Radiobutton(self.root, text="Binary", variable=mode,
                    value=BINARY).pack(side='left', pady=4)
        Radiobutton(self.root, text="Edge", variable=mode,
                    value=EDGE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Line", variable=mode,
                    value=LINE).pack(side='left', pady=4)
        Radiobutton(self.root, text="Abs Diff", variable=mode,
                    value=ABSDIFF).pack(side='left', pady=4)
        Radiobutton(self.root, text="RGB", variable=mode,
                    value=RGB).pack(side='left', pady=4)
        Radiobutton(self.root, text="HSV", variable=mode,
                    value=HSV).pack(side='left', pady=4)
        Radiobutton(self.root, text="Corners", variable=mode,
                    value=CORNERS).pack(side='left', pady=4)
        Radiobutton(self.root,
                    text="Find Contours",
                    variable=mode,
                    value=CONTOURS).pack(side='left', pady=4)

        # threading
        self.stopevent = threading.Event()
        self.thread = threading.Thread(target=self.capture, args=())
        self.thread.start()

    def capture(self):
        while not self.stopevent.is_set():
            if not self.stopflag:
                ret0, frame = camera.read()
                if mode.get() == BINARY:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    ret, lower = cv.threshold(grey, lThreshold, 255,
                                              cv.THRESH_BINARY)
                    ret, upper = cv.threshold(grey, hThreshold, 255,
                                              cv.THRESH_BINARY_INV)
                    frame = cv.bitwise_and(lower, upper)
                elif mode.get() == EDGE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    blurred = cv.GaussianBlur(grey, (5, 5), 0)
                    frame = cv.Canny(blurred, lThreshold * 2, hThreshold * 2)
                elif mode.get() == LINE:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    # grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    blurred = cv.GaussianBlur(frame, (5, 5), 0)
                    edges = cv.Canny(blurred, 100, 200)
                    lines = cv.HoughLines(edges, 1, np.pi / 180, lThreshold,
                                          None, 0, 0)
                    if lines is not None:
                        for i in range(0, len(lines)):
                            rho = lines[i][0][0]
                            theta = lines[i][0][1]
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                            cv.line(frame, pt1, pt2, (0, 0, 255), 3,
                                    cv.LINE_AA)

                elif mode.get() == ABSDIFF:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    tmp = frame.copy()
                    frame = cv.absdiff(frame, self.buffer)
                    self.buffer = tmp
                elif mode.get() == RGB:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                elif mode.get() == HSV:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    # Add your code here
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                elif mode.get() == CORNERS:
                    lThreshold = Slider1.get() + 1
                    hThreshold = Slider2.get() + 1
                    # Add your code here
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    # Parameters for Shi-Tomasi algorithm
                    qualityLevel = 0.01
                    minDistance = 10
                    blockSize = 3
                    gradientSize = 3
                    useHarrisDetector = False
                    k = 0.04

                    # Apply corner detection
                    corners = cv.goodFeaturesToTrack(grey, 100, np.clip(lThreshold/255,.001,.999), np.clip(hThreshold, .001, .999), None, \
                    blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
                    # Set the needed parameters to find the refined corners
                    winSize = (5, 5)
                    zeroZone = (-1, -1)
                    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT,
                                40, 0.001)
                    # Calculate the refined corner locations
                    corners = cv.cornerSubPix(grey, corners, winSize, zeroZone,
                                              criteria)
                    # Draw corners detected
                    print('** Number of corners detected:', corners.shape[0])
                    radius = 4
                    for i in range(corners.shape[0]):
                        cv.circle(
                            frame,
                            (int(corners[i, 0, 0]), int(corners[i, 0, 1])),
                            radius, (0, 0, 255), cv.FILLED)

                elif mode.get() == CONTOURS:
                    lThreshold = Slider1.get()
                    hThreshold = Slider2.get()
                    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                    blurred = cv.GaussianBlur(grey, (15, 15), 0)
                    edges = cv.Canny(blurred, 0* 5, 9 * 5)

                    # maybe some dilation and erosion
                    kernel = np.ones((3,3),np.uint8)
                    edges = cv.dilate(edges,kernel,iterations = 10)
                    edges = cv.erode(edges,kernel,iterations = 10)

                    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    for i in range(len(contours)):
                        color = (0,0,255)
                        rect = cv.boundingRect(contours[i])
                        x,y,w,h = rect
                        hull = cv.convexHull(contours[i])
                        # area of convex hull
                        area_hull = cv.contourArea(hull)
                        if area_hull > 10*100:
                            #calculate the convex hull of each contour

                            cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                            # cv.drawContours(frame, contours, i, color, 2, cv.LINE_8, hierarchy, 0)

                image = cvMat2tkImg(frame)
                self.panel.configure(image=image)
                self.panel.image = image
                videoout.write(frame)

    def startstop(self):  # toggle flag to start and stop
        if btnStart.config('text')[-1] == 'Start':
            btnStart.config(text='Stop')
        else:
            btnStart.config(text='Start')
        self.stopflag = not self.stopflag

    def run(self):  # run main loop
        self.root.mainloop()

    def quit(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(
            1.0, self.stop
        )  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def exitApp(self):  # exit loop
        self.stopflag = True
        t = threading.Timer(
            1.0, self.stop
        )  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def stop(self):
        self.stopevent.set()
        self.root.quit()


app = App()
app.run()
# release the camera
camera.release()
