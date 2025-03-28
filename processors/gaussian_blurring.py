import cv2
import numpy as np

def gaussian_blurring(image, kernal_size=3, sigmaX=0):

    kernal_size = (kernal_size, kernal_size)
    blurred = cv2.GaussianBlur(image, kernal_size, sigmaX)
    return blurred