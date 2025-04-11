import gradio as gr
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO('best.pt')

class AnnotationState:
    def __init__(self):
        self.boxes = []
        self.labels = []
        self.confidences = []
        self.selected_box = None
        self.class_names = model.names
        self.current_image = None
        # Initialize available classes
        self.available_classes = list(model.names.values())

state = AnnotationState()

def process_image(image):
    if image is None:
        return None, [], [], [], []
    
    state.current_image = image
    
    # Convert to numpy array if needed
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = np.array(image)
    
    # Run inference
    results = model(image)[0]
    
    # Get detection results
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    # Update state
    state.boxes = boxes.tolist()
    state.labels = [model.names[i] for i in class_ids]
    state.confidences = confidences.tolist()
    
    return draw_annotations(image), state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes)

def draw_annotations(image):
    if image is None:
        return None
    
    annotated_image = image.copy()
    
    # Draw existing boxes
    for i, (box, label, conf) in enumerate(zip(state.boxes, state.labels, state.confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if i != state.selected_box else (255, 0, 0)
        thickness = 3 if i == state.selected_box else 2
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, thickness)
        label_text = f"{label} {conf:.2f}"
        cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_image

def select_box(image, evt: gr.SelectData):
    if state.current_image is None or not state.boxes:
        return image, state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes), 0, 0, 0, 0
    
    x, y = evt.index
    # Check if clicked on existing box
    for i, box in enumerate(state.boxes):
        x1, y1, x2, y2 = map(int, box)
        if x1 <= x <= x2 and y1 <= y <= y2:
            state.selected_box = i
            # Return the selected box data
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

def add_box(image, evt: gr.SelectData):
    if state.current_image is None:
        return image, state.boxes, state.labels, state.confidences, gr.Dropdown(choices=state.available_classes), 0, 0, 0, 0
    
    x, y = evt.index
    # Add new box around click point with default size
    box_size = 50
    x1 = max(0, x - box_size//2)
    y1 = max(0, y - box_size//2)
    x2 = min(image.shape[1], x + box_size//2)
    y2 = min(image.shape[0], y + box_size//2)
    
    state.boxes.append([x1, y1, x2, y2])
    state.labels.append(state.available_classes[0])  # Use first available class as default
    state.confidences.append(1.0)
    state.selected_box = len(state.boxes) - 1
    
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

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Interactive Object Detection Annotation Tool")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            run_button = gr.Button("Run Detection")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results (Click to select/add boxes)")
            
            with gr.Row():
                delete_btn = gr.Button("Delete Selected Box")
                add_btn = gr.Button("Add Box at Click")
            
            with gr.Row():
                class_dropdown = gr.Dropdown(label="Class", choices=state.available_classes, interactive=True)
            
            with gr.Row():
                x1_input = gr.Number(label="X1", precision=0)
                y1_input = gr.Number(label="Y1", precision=0)
                x2_input = gr.Number(label="X2", precision=0)
                y2_input = gr.Number(label="Y2", precision=0)
                update_coords_btn = gr.Button("Update Coordinates")
            
            boxes_output = gr.JSON(label="Bounding Boxes")
            labels_output = gr.JSON(label="Labels")
            conf_output = gr.JSON(label="Confidences")
    
    # Set up event handlers
    run_button.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, boxes_output, labels_output, conf_output, class_dropdown]
    )
    
    output_image.select(
        fn=select_box,
        inputs=[output_image],
        outputs=[output_image, boxes_output, labels_output, conf_output, class_dropdown, x1_input, y1_input, x2_input, y2_input]
    )
    
    add_btn.click(
        fn=add_box,
        inputs=[output_image],
        outputs=[output_image, boxes_output, labels_output, conf_output, class_dropdown, x1_input, y1_input, x2_input, y2_input]
    )
    
    delete_btn.click(
        fn=delete_selected,
        outputs=[output_image, boxes_output, labels_output, conf_output, class_dropdown, x1_input, y1_input, x2_input, y2_input]
    )
    
    class_dropdown.change(
        fn=update_selected_class,
        inputs=[class_dropdown],
        outputs=[output_image, boxes_output, labels_output, conf_output]
    )
    
    update_coords_btn.click(
        fn=update_box_coordinates,
        inputs=[x1_input, y1_input, x2_input, y2_input],
        outputs=[output_image, boxes_output, labels_output, conf_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch() 