import cv2
import numpy as np
def generate_reflection_mask(reflection_image, kernal_size=3, threshold=220):
    _, mask = cv2.threshold(reflection_image, threshold, 255, cv2.THRESH_BINARY)
    
    kernal = (kernal_size, kernal_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    return mask