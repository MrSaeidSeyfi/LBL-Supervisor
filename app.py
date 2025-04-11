import gradio as gr
from app.state import state
from app.handlers import (
    process_image, select_box, create_label,
    delete_selected, update_selected_class, 
    update_box_coordinates
)

with gr.Blocks() as app:
    gr.Markdown("# Interactive Object Detection Annotation Tool")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="numpy")
            run_button = gr.Button("Run Detection")
        
        with gr.Column():
            output_image = gr.Image(label="Detection Results (Click to select boxes)")
            
            with gr.Row():
                delete_btn = gr.Button("Delete Selected Box")
                create_btn = gr.Button("Create Label")
            
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
    
    # Select box when clicking on the image
    output_image.select(
        fn=select_box,
        inputs=[output_image],
        outputs=[output_image, boxes_output, labels_output, conf_output, class_dropdown, x1_input, y1_input, x2_input, y2_input]
    )
    
    # Create a new label with the selected class
    create_btn.click(
        fn=create_label,
        inputs=[class_dropdown],
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

if __name__ == "__main__":
    app.launch() 