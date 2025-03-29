
from .remove_reflection import remove_reflection_test as remove_reflection
from .gamma_correction import gamma_correction
from .laplacian_filtering import laplacian_filter as laplacian_filter
from .laplacian_filtering import rgb_laplacian_filter as rgb_laplacian_filter
from .histogram_equalization import *
from .edge_detection import *
from .morphologyEx import *

import cv2
import os
import numpy as np
import sys

def save_image(image, path):
    output_path = os.path.join("outputs", path+".jpg")
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

def process_image(image_path):
    def wrapper(func):
        def run(*args, **kwargs):
            processed_image = func(*args, **kwargs)
            save_image(processed_image, image_path)
            return processed_image
        return run
    return wrapper


#a decorator for show a window with side bar 
#can auto find all local vaiables of the function name start with "param_"
#capture local variables and create trackbars for them
#so that user can change the parameters in real time
def testing(func):
    def wrapper(*args, **kwargs):
        captured_locals = {}
        current_param = None
        input_value = ''

        # Tracer function to capture locals
        def tracer(frame, event, arg):
            if event == "return":
                captured_locals.update(frame.f_locals.copy())
            return tracer

        # Initial function execution
        sys.settrace(tracer)
        try:
            image = func(*args, **kwargs)
        finally:
            sys.settrace(None)

        params = {k: v for k, v in captured_locals.items() if k.startswith("param_")}
        param_list = list(params.keys())
        
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", 800, 600)

        def update_display():
            nonlocal image
            # Create display image with parameter info
            display = image.copy()
            y = 30
            for i, (name, value) in enumerate(params.items()):
                color = (0, 255, 0) if param_list[i] == current_param else (255, 255, 255)
                cv2.putText(display, f"{i+1}. {name}: {value}", (10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y += 30
            if current_param is not None:
                cv2.putText(display, f"New value for {current_param}: {input_value}", 
                           (10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Image", cv2.resize(display, (800, 600)))

        update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF
        
            if key == 27:  # ESC to exit
                break
                
            if current_param is None:
                # Select parameter using number keys
                if 49 <= key <= 57:  # 1-9 keys
                    idx = key - 49
                    if idx < len(param_list):
                        current_param = param_list[idx]
                        input_value = ''
            else:
                # Handle numeric input
                if 48 <= key <= 57:  # 0-9
                    input_value += chr(key)
                elif key == 13:  # Enter to confirm
                    if input_value:
                        new_val = min(max(int(input_value), 0), 255)
                        captured_locals[current_param] = new_val
                        params[current_param] = new_val
                        
                        # Re-run function with updated params
                        def update_tracer(frame, event, arg):
                            if event == "line":
                                frame.f_locals[current_param] = new_val
                            return update_tracer
                        
                        sys.settrace(update_tracer)
                        image = func(*args, **kwargs)
                        sys.settrace(None)
                    current_param = None
                    input_value = ''
                elif key == 8:  # Backspace
                    input_value = input_value[:-1]

            update_display()

        cv2.destroyAllWindows()
        return image

    return wrapper