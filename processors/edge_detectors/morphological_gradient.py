import cv2
import numpy as np
from ..utils.display import display

def morphological_gradient(
    img, 
    ksize: int = 3,
    normalize: bool = True,
    show_result: bool = False,
    window_size: tuple[int, int] = (10, 6),
    dpi: int = 100 
) -> np.ndarray:

    #if len(img.shape) == 3:
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((ksize, ksize), np.uint8)
    edges = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    edges = np.abs(edges)

    if normalize:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    edges = edges.astype(np.uint8)
    
    if show_result:
        display(edges, title='morphological_gradient', window_size=window_size, dpi=dpi)
    return edges