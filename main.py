import cv2
import os
from processors import remove_reflection
from processors import gamma_correction
from processors import save_image, process_image
from processors import laplacian_filter


# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="laplacian_filter.jpg")
def main(image):
    processed_image = processed_image = laplacian_filter(image, 3)
    return processed_image

if __name__ == "__main__":
    main(image)