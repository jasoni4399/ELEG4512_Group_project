
from .remove_reflection import remove_reflection_test as remove_reflection
from .gamma_correction import gamma_correction
from .laplacian_filtering import laplacian_filter as laplacian_filter
from .laplacian_filtering import rgb_laplacian_filter as rgb_laplacian_filter
from .histogram_equalization import *
from .edge_detection import *
from .morphologyEx import *

import cv2
import os

def save_image(image, path):
    output_path = os.path.join("outputs", path+".jpg")
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

def process_image(image_path):
    def wrapper(func):
        def run(*args, **kwargs):
            processed_image = func(*args, **kwargs)
            save_image(processed_image, image_path)
            return processed_image
        return run
    return wrapper

#a decorator for show a window with side bar 
#to test every variable using in the function that 
#start with "param_" for testing
def testing(func):
    def wrapper(*args, **kwargs):
        # Create a window with a trackbar for each parameter
        cv2.namedWindow("Image")
        params = {k: v for k, v in kwargs.items() if k.startswith("param_")}
        for param in params:
            cv2.createTrackbar(param, "Image", 0, 255, lambda x: None)

        # Call the function and display the image
        image = func(*args, **kwargs)
        cv2.imshow("Image", image)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break

        cv2.destroyAllWindows()
        return image
    return wrapper