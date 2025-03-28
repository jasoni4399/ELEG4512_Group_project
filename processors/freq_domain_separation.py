import cv2
import numpy as np

def frequency_domain_separation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1
    
    fshift = dft_shift * (1 - mask)
    f_ishift = np.fft.ifftshift(fshift)
    reflection = cv2.idft(f_ishift)[:,:,0]
    
    reflection = cv2.normalize(reflection, None, 0, 255, cv2.NORM_MINMAX)
    return reflection.astype(np.uint8)