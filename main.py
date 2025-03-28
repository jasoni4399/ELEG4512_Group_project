import cv2
import os
import numpy as np
from processors import *

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="test")#file name to save the image
def main(image):
    processed_image = image.copy()
    for i in range(5):
        temp = gamma_correction(processed_image, gamma=20)
        processed_image = cv2.addWeighted(processed_image, 0.5, temp, 0.5, 0)
    processed_image = rgb_histogram_equalization(processed_image)
    #image - processed_image
    

    return image - processed_image

if __name__ == "__main__":
    main(image)