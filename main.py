import cv2
import os
from processors import remove_reflection
from processors import gamma_correction
from processors import process_image

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="gamma_correction_10")#file name to save the image
def main(image):
    processed_image = processed_image = gamma_correction(image, 10)
    return processed_image

if __name__ == "__main__":
    main(image)