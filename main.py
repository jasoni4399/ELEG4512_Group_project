import cv2
import os
import numpy as np
from processors import remove_reflection
from processors import gamma_correction, laplacian_filter, rgb_laplacian_filter
from processors import process_image,save_image

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="laplacian_3x3_color")#file name to save the image
def main(image):
    
    result= rgb_laplacian_filter(image, kernel_size=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT)
    image -= 5*result
    return image

if __name__ == "__main__":
    main(image)