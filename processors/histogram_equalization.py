#histogram equalization
import cv2
import numpy as np
import os

def histogram_equalization(image):
    if len(image.shape) == 3:
        #RGB image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    equalized = cv2.equalizeHist(gray)
    
    return equalized

def rgb_histogram_equalization(image):
    b, g, r = cv2.split(image)
    
    b_eq = histogram_equalization(b)
    g_eq = histogram_equalization(g)
    r_eq = histogram_equalization(r)
    
    return cv2.merge((b_eq, g_eq, r_eq))