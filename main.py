import cv2
import os
import numpy as np
from processors import *

# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="test")#file name to save the image
def main(image):
    #k means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    # apply kmeans clustering
    compactness,labels,centers = cv2.kmeans(Z, 5, None, criteria, 10, flags)
    print("compactness: ", compactness)
    print("labels: ", labels)
    print("centers: ", centers)
    center = np.uint8(center)
    res = center[labels.flatten()]
    res2 = res.reshape((image.shape))
    return res2
 
if __name__ == "__main__":
    main(image)