import cv2
import os
import numpy as np
from processors import *

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)
show_result = False

@process_image(image_path="edge_detection_outputs/sobel/sobel_5_1_1_False_False")#file name to save the image
def main(image):

    output = sobel(image, ksize=5, dx=1, dy=1, normalize=False, uint8=False, show_result=True)
    #output = laplacian(image, ksize=3, normalize=False, uint8=True, show_result=True)
    #output = LoG(image, ksize=3, sigma=1.0, normalize=False, show_result=True)
    #output = canny(image, threshold1=100, threshold2=300, apertureSize=3, normalize=False, show_result=True)
    #output = morphologyEx(image, ksize=5, normalize=False, show_result=True)
    #output = fft(image, radius=400, normalize=False, show_result=True)

    return output

if __name__ == "__main__":
    main(image)