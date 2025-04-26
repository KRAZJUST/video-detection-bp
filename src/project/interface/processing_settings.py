from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QCheckBox, QDialogButtonBox

class ProcessingSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Settings")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout for settings
        form_layout = QFormLayout()
                
        # Checkbox for using segmentation when processing with YOLO
        self.use_segmentation_checkbox = QCheckBox()
        self.use_segmentation_checkbox.setChecked(parent.processing_segmentation_value if hasattr(parent, 'processing_segmentation_value') else False)
        self.use_segmentation_checkbox.setToolTip("Use segmentation masks for calculation of " \
                                                "the dominant color of the detected objects. \n" \
                                                "This feature will make the color calculation more precise \n" \
                                                "but will take a longer time to process." \
                                                "\n\n" \
                                                "Note: This feature is availible only for YOLO and ByteTrack.\n" \
                                                "Tip: This features is not recommended to use with ByteTrack \n" \
                                                "as the ByteTrack has averaged color calculation even without segmentation \n" \
                                                "and the segmentation will take a lot of time to process.")
        form_layout.addRow("Use Segmentation:", self.use_segmentation_checkbox)
        # Checkbox for skiping the siglip frames based on the yolo detections
        self.skip_siglip_frames_checkbox = QCheckBox()
        self.skip_siglip_frames_checkbox.setChecked(parent.skip_siglip_with_yolo_value if hasattr(parent, 'skip_siglip_with_yolo_value') else False)
        self.skip_siglip_frames_checkbox.setToolTip("Skip the frames being processed with SigLIP based on the YOLO detections. \n" \
                                                "This feature will make the processing faster and possibly more precise \n" \
                                                "but only on the videos with a less crowded scenes.\n" \
                                                "If the scene has a lot of objects, the processing will take approximately 1.05-1.1 times longer. \n")
        form_layout.addRow("Skip SigLIP with YOLO:", self.skip_siglip_frames_checkbox)
        layout.addLayout(form_layout)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
       