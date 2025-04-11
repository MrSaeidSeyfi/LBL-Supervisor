import cv2
import numpy as np
from PIL import Image
from app.state import state

def draw_annotations(image):
    if image is None:
        return None
    
    annotated_image = image.copy()
    
    for i, (box, label, conf) in enumerate(zip(state.boxes, state.labels, state.confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if i != state.selected_box else (255, 0, 0)
        thickness = 3 if i == state.selected_box else 2
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{label} {conf:.2f}"
        cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_image

def prepare_image(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    return image 