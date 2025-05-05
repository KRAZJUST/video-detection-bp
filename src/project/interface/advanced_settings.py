# =============================================================================
# File: advanced_settings.py
# Author: David Skalka (xskalk03@stud.fit.vutbr.cz)
# Faculty of Information Technology, Brno University of Technology
# Academic Year: 2024/2025
#
# This file is part of the bachelor's thesis:
# "Recognizing people and their activities in video from security cameras"
#
# Description:
# This module provides a dialog for advanced settings when querying for 
# the results in the application UI.
#
# =============================================================================

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QSpinBox, QCheckBox, QDialogButtonBox

class AdvancedSettings(QDialog):
    """
    Dialog for advanced settings when querying for results.
    This dialog allows the user to set parameters for XCLIP batch count,
    SigLIP frame count, deduplication of frames, and segmentation usage.
    """
    def __init__(self, parent=None):
        """
        Initialize the dialog with the parent widget.
        Args:
            parent: Parent widget
        """
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