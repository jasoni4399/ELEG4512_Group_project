import cv2
import numpy as np
from ..utils.display import display

def fourier_highpass_filter(
    img, 
    radius: int = 30,
    normalize: bool = True,
    show_result: bool = False,
    window_size: tuple[int, int] = (10, 6),
    dpi: int = 100 
):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 0, -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    edges = np.fft.ifft2(f_ishift)
    edges = np.abs(edges)

    if normalize:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    if show_result:
        display(edges, title='fourier_highpass_filter', normalize=normalize, window_size=window_size, dpi=dpi)
    
    return edges