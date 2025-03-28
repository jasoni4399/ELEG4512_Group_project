import cv2
import os
import numpy as np
from processors import *

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)
show_result = False

@process_image(image_path="edge_detection_outputs/fft_400")#file name to save the image
def main(image):

    #output = sobel(image, ksize=5, dx=1, dy=1, normalize=True, show_result=show_result)
    #output = laplacian(image, ksize=5, normalize=True, show_result=show_result)
    #output = LoG(image, ksize=5, sigma=5, normalize=False, show_result=show_result)
    #output = canny(image, threshold1=100, threshold2=300, apertureSize=3, normalize=True, show_result=show_result)
    #output = morphologyEx(image, ksize=5, normalize=True, show_result=show_result)
    output = fft(image, radius=400, normalize=False, show_result=show_result)

    return output

if __name__ == "__main__":
    main(image)