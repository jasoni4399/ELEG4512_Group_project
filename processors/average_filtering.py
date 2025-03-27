import cv2
import numpy as np

def average_filter(image, kernel_size=3):

    kernel_size = (kernel_size, kernel_size)
    
    blurred = cv2.blur(image, kernel_size)
    return blurred