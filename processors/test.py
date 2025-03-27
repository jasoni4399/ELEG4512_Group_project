import cv2
import numpy as np
import os

def remove_reflection_test(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result

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
