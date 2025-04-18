from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QSpinBox, QCheckBox, QDialogButtonBox

class AdvancedSettings(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Query Settings")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout for settings
        form_layout = QFormLayout()
                
        # XCLIP batch count
        self.batch_count_spinbox = QSpinBox()
        self.batch_count_spinbox.setRange(1, 100)
        self.batch_count_spinbox.setValue(parent.results_batch_count if hasattr(parent, 'results_frames_count') else 5)
        form_layout.addRow("XCLIP Batch Count:", self.batch_count_spinbox)
        
        # SigLIP frame count
        self.frame_count_spinbox = QSpinBox()
        self.frame_count_spinbox.setRange(1, 1000)
        self.frame_count_spinbox.setValue(parent.results_frames_count if hasattr(parent, 'results_frames_count') else 50)
        form_layout.addRow("SigLIP Frame Count:", self.frame_count_spinbox)
        
        # Add batch mode checkbox for siglip
        self.batch_mode_checkbox = QCheckBox()
        self.batch_mode_checkbox.setChecked(parent.batch_mode_value if hasattr(parent, 'batch_mode_value') else False)
        self.batch_mode_checkbox.setToolTip("Use batch mode for SigLIP. \n")
        form_layout.addRow("Batch Mode:", self.batch_mode_checkbox)

        # Add deduplication checkbox
        self.deduplicate_frames_checkbox = QCheckBox()
        self.deduplicate_frames_checkbox.setChecked(parent.deduplicate_frames_value if hasattr(parent, 'deduplicate_frames_value') else True)
        self.deduplicate_frames_checkbox.setToolTip("Deduplicate frames with identical objects so that \n"
                                                  "it is easier to navigate through the results.")
        form_layout.addRow("Deduplicate Frames:", self.deduplicate_frames_checkbox)
        
        # Add segmentation checkbox
        self.use_segmentation_checkbox = QCheckBox()
        self.use_segmentation_checkbox.setChecked(parent.use_segmentation_value if hasattr(parent, 'use_segmentation_value') else False)
        self.use_segmentation_checkbox.setToolTip("Use segmentation masks for object detection. \n"
                                        "This feature will not make the search more precise \n"
                                        "but will display the found objects more accurately \n"
                                        "at the cost of performance.")
        form_layout.addRow("Use Segmentation:", self.use_segmentation_checkbox)
        
        layout.addLayout(form_layout)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)