import numpy as np

def erosion(img, kernel):
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")
    
    h, w = img.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.min(patch * kernel)
    return output