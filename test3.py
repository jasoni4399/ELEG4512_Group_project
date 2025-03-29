import cv2
import os
import numpy as np
from processors import *
# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="hough_tranform")#file name to save the image
def main(image):
    #hough transform to detect lines
    b,g, r = cv2.split(image)
    #find b histogram for threshold
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    #find the peak of the histogram
    b_peak = np.argmax(b_hist)
    g_peak = np.argmax(g_hist)
    r_peak = np.argmax(r_hist)
    #find the threshold for the histogram
    b_threshold = b_peak + 50
    g_threshold = g_peak + 50
    r_threshold = r_peak + 50
    print("b_peak: ", b_peak, "g_peak: ", g_peak, "r_peak: ", r_peak)
    print("b_threshold: ", b_threshold, "g_threshold: ", g_threshold, "r_threshold: ", r_threshold)
    #threshold the image
    _, b = cv2.threshold(b, b_threshold, 255, cv2.THRESH_BINARY)
    _, g = cv2.threshold(g, g_threshold, 255, cv2.THRESH_BINARY)
    _, r = cv2.threshold(r, r_threshold, 255, cv2.THRESH_BINARY)
    #find the edges of the image by using the threshold for canny and test the threshold of the histogram
    b_edges = cv2.Canny(b, 50, 150)
    #find the lines of the image by using the hough transform
    b_lines = cv2.HoughLinesP(b_edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    g_lines = cv2.HoughLinesP(g_edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    r_lines = cv2.HoughLinesP(r_edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    # Draw lines on the image
    if b_lines is not None:
        for line in b_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if g_lines is not None:
        for line in g_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if r_lines is not None:
        for line in r_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image

if __name__ == "__main__":
    main(image)