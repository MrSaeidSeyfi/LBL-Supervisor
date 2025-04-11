import gradio as gr
from app.state import state
from app.model import model
from app.utils import draw_annotations, prepare_image
import cv2

def process_image(image):
    if image is None:
        return None, [], [], [], []
    
    state.current_image = image
    image = prepare_image(image)
    results = model(image)[0]
    
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    state.boxes = boxes.tolist()
    state.labels = [model.names[i] for i in class_ids]
    state.confidences = confidences.tolist()
    
    return draw_annotations(image), state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes)

def select_box(image, evt: gr.SelectData):
    if state.current_image is None or not state.boxes:
        return image, state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes), 0, 0, 0, 0
    
    x, y = evt.index
    for i, box in enumerate(state.boxes):
        x1, y1, x2, y2 = map(int, box)
        if x1 <= x <= x2 and y1 <= y <= y2:
            state.selected_box = i
            return (
                draw_annotations(state.current_image), 
                state.boxes, 
                state.labels, 
                state.confidences,
                gr.Dropdown(choices=state.available_classes, value=state.labels[i]),
                x1, y1, x2, y2
            )
    
    state.selected_box = None
    return draw_annotations(state.current_image), state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes), 0, 0, 0, 0

def create_label(selected_class):
    if state.current_image is None:
        return None, [], [], [], gr.Dropdown(choices=state.available_classes), 0, 0, 0, 0
    
    # Create a new box in the center of the image with default size
    height, width = state.current_image.shape[:2]
    box_size = 100
    
    x1 = max(0, width // 2 - box_size // 2)
    y1 = max(0, height // 2 - box_size // 2)
    x2 = min(width, width // 2 + box_size // 2)
    y2 = min(height, height // 2 + box_size // 2)
    
    # Add new box
    state.boxes.append([x1, y1, x2, y2])
    
    # Use selected class or default to first available
    if selected_class and selected_class in state.available_classes:
        state.labels.append(selected_class)
    else:
        default_class = state.available_classes[0] if state.available_classes else "object"
        state.labels.append(default_class)
    
    state.confidences.append(1.0)
    
    # Set as selected box
    state.selected_box = len(state.boxes) - 1
    
    # Return updated UI
    return (
        draw_annotations(state.current_image), 
        state.boxes, 
        state.labels, 
        state.confidences,
        gr.Dropdown(choices=state.available_classes, value=state.labels[-1]),
        x1, y1, x2, y2
    )

def delete_selected():
    if state.selected_box is not None:
        del state.boxes[state.selected_box]
        del state.labels[state.selected_box]
        del state.confidences[state.selected_box]
        state.selected_box = None
    
    return draw_annotations(state.current_image), state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes), 0, 0, 0, 0

def update_selected_class(new_class):
    if state.selected_box is not None and new_class in state.available_classes:
        state.labels[state.selected_box] = new_class
    
    return draw_annotations(state.current_image), state.boxes, state.labels, state.confidences

def update_box_coordinates(x1, y1, x2, y2):
    if state.selected_box is not None:
        try:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            state.boxes[state.selected_box] = [x1, y1, x2, y2]
        except (ValueError, TypeError):
            pass
    
    return draw_annotations(state.current_image), state.boxes, state.labels, state.confidences 