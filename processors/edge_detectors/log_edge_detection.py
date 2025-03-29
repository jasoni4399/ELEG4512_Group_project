import cv2
import numpy as np
from ..utils.display import display

def log_edge_detection(
    img, 
    ksize: int = 5, 
    sigma: float = 1.0, 
    normalize: bool = True,
    show_result: bool = False,
    window_size: tuple[int, int] = (10, 6),
    dpi: int = 100 
) -> np.ndarray:

    #if len(img.shape) == 3:
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    edges = cv2.Laplacian(blurred, cv2.CV_64F)
    edges = np.absolute(edges)
    
    if normalize:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    edges = edges.astype(np.uint8)

    if show_result:
        display(edges, title='Log Edge Detection', window_size=window_size, dpi=dpi)

    return edges