import cv2
import numpy as np
from ..utils.display import display

def sobel_edge_detection(
    img: np.ndarray,
    ksize: int = 3,
    dx: int = 1,
    dy: int = 1,
    normalize: bool = True,
    show_result: bool = False,
    window_size: tuple[int, int] = (10, 6),
    dpi: int = 100 
) -> np.ndarray:
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sobel_x = cv2.Sobel(img, cv2.CV_64F, dx, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, dy, ksize=ksize)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    if normalize:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    if show_result:
        display(edges, title='Sobel Edge Detection', window_size=window_size, dpi=dpi)
        
    return edges