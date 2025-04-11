from app.model import model

class AnnotationState:
    def __init__(self):
        self.boxes = []
        self.labels = []
        self.confidences = []
        self.selected_box = None
        self.class_names = model.names
        self.current_image = None
        self.available_classes = list(model.names.values())
        self.drawing_mode = False
        self.drawing_start = None

state = AnnotationState() 