import numpy as np

def dilation(img, kernel):

    if len(img.shape) != 2:
        raise ValueError("Input image must be 2D (grayscale or binary)")
    
    h, w = img.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            window = padded_img[i:i+k_h, j:j+k_w]
            output[i, j] = np.max(window * kernel)
    
    return output