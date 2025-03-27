import cv2
import numpy as np

def gamma_correction(image, gamma=1.0, c=1.0):

    normalized = image.astype(np.float32) / 255.0
    result = c * np.power(normalized, gamma)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return result