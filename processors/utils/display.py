import matplotlib.pyplot as plt
import cv2
import numpy as np

def display(img,  
            title: str = 'Image',
            window_size: tuple[int, int] = (10, 6),
            dpi: int = 100) -> None:
    plt.figure(figsize=window_size, dpi=dpi)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()