import cv2
import os
import numpy as np
from processors import remove_reflection
from processors import *
from processors import process_image,save_image

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="adaptive_histogram_equalization")#file name to save the image
def main(image):
    processed_image = rgb_adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8))
    return processed_image

if __name__ == "__main__":
    main(image)