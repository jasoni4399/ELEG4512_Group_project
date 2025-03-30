__all__ = ["save_image", "process_image","testing","gamma_correction", "gamma_clahe", "generate_reflection_mask", 
           "edge_based_segmentation", "filter_draw_contours", "display"] 

from .contrast.gamma_correction import gamma_correction
from .contrast.gamma_clahe import gamma_clahe

from .reflection.reflection_mask import generate_reflection_mask

from .segmentation.edge_based_segmentation import edge_based_segmentation

from .utils.filter_draw_contours import  filter_draw_contours
from .utils.display import display


import cv2
import os
import sys

def save_image(image, path):
    output_path = os.path.join("outputs", path+".png")
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
        # Set the function to be traced
        # Get the local variables of the function
        captured_locals = {}  # Use a mutable object to avoid scoping issues

        # Define the tracer function
        def tracer(frame, event, arg):
            if event == "return":
                # Capture the locals when the function returns
                captured_locals.update(frame.f_locals.copy())
            return tracer  # Return itself to keep tracing active

        # Activate the tracer
        sys.settrace(tracer)
        try:
            image = func(*args, **kwargs)
        finally:
            sys.settrace(None)  # Ensure tracing is disabled
        # Find parameters starting with "param_"
        params = {k: v for k, v in captured_locals.items() if k.startswith("param_")}
        # Create a window with a trackbar for each parameter
        cv2.namedWindow("Image",0)
        for param in params:
            # Create a trackbar for each parameter
            # The trackbar name is the parameter name without "param_"
            trackbar_name = param.replace("param_", "")
            cv2.createTrackbar(trackbar_name, "Image", params[param], 255, lambda x: None)
            # Set the initial value of the trackbar to the parameter value
            cv2.setTrackbarPos(trackbar_name, "Image", params[param])

        current_width = cv2.getWindowImageRect("Image")[2]
        current_height = cv2.getWindowImageRect("Image")[3]
        image_resized = cv2.resize(image, (current_width, current_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Image", image_resized)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to exit
                break
            # Update the parameters based on the trackbar positions
            updated = False
            for param in params:
                trackbar_name = param.replace("param_", "")
                new_val = cv2.getTrackbarPos(trackbar_name, "Image")
                if new_val != captured_locals[param]:
                    captured_locals[param] = new_val
                    updated = True
            # Re-run the function if parameters changed
            if updated:
                # Update the function's variables with the new parameter values
                def update_tracer(frame, event, arg):
                    if event == "line":
                        for param in params:
                            # Update the parameter value in the function's local scope
                            frame.f_locals[param] = captured_locals[param]
                    return update_tracer
                sys.settrace(update_tracer)
                image = func(*args, **kwargs)
                sys.settrace(None)
                #resize the image to fit the window
                current_width = cv2.getWindowImageRect("Image")[2]
                current_height = cv2.getWindowImageRect("Image")[3]
                image_resized = cv2.resize(image, (current_width, current_height), interpolation=cv2.INTER_AREA)
                cv2.imshow("Image", image_resized)
        # Cleanup
        cv2.destroyAllWindows()
        return image
    return wrapper