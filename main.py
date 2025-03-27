import cv2
import os
import numpy as np
from processors import laplacian_filter, histogram_normalization, contrast_stretching, average_filter
from processors import process_image

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="laplacian_5x5_reversed")#file name to save the image
def main(image):
    processed_image = average_filter(image, 9)
    
    return processed_image

if __name__ == "__main__":
    main(image)