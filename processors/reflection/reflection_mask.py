import cv2
import numpy as np
def generate_reflection_mask(reflection_image, kernel_size=3, threshold=220):
    _, mask = cv2.threshold(reflection_image, threshold, 255, cv2.THRESH_BINARY)
    
    kernal = (kernel_size, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(mask, (5,5), 0)
    
    # Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)
    gradient = cv2.magnitude(sobelx, sobely)
    gradient_enhanced = np.power(gradient / gradient.max(), 0.5) * 255
    gradient_enhanced = gradient_enhanced.astype(np.uint8)

    # Gaussian blurring
    blurred = cv2.GaussianBlur(gradient_enhanced, (5,5), 0)

    # Canny
    edges = cv2.Canny(blurred, 400, 700)

    # Gaussian blurring
    blurred = cv2.GaussianBlur(edges, (3,3), 0)

    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    
    return mask