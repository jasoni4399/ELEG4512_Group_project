import cv2
import numpy as np
from processors import *
# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@testing
def main(image):
    #use image as weighted mask for the original image
    param_weighted = 1
    param_weight_base=10
    #translate the weighted mask to the image
    param_x = 1
    param_translatey = 0
    #create the weighted mask
    weidhted_mask = image.copy()
    weidhted_mask = cv2.multiply(weidhted_mask, param_weighted/param_weight_base)
    #translate the weidhted_mask by param_translat_x and param_translat_y
    print(image.shape)
    height, width = image.shape[:2] 
    print("translate_x: ", 255*param_x, "translate_y: ", param_translatey*height/255)
    T = np.float32([[1, 0, param_x*255], [0, 1, param_translatey*height/255]]) 
    translated_weighted_image = cv2.warpAffine(weidhted_mask,T,(width, height))
    #mask the image with the weighted mask
    masked_image = cv2.addWeighted(image, 1, translated_weighted_image, -1, 0)
    return translated_weighted_image

if __name__ == "__main__":
    main(image)
