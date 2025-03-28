import cv2
import numpy as np
def multi_scale_inpainting(image, mask, scale_factor=0.25):

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape[:2]
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    small_img = cv2.resize(image, (new_w, new_h))
    small_mask = cv2.resize(mask, (new_w, new_h))
    
    _, small_mask = cv2.threshold(small_mask, 127, 255, cv2.THRESH_BINARY)
    
    inpainted_small = cv2.inpaint(
        small_img, 
        small_mask, 
        inpaintRadius=3, 
        flags=cv2.INPAINT_NS
    )
    
    inpainted = cv2.resize(inpainted_small, (w, h))
    return inpainted