"""
End-to-end inference pipeline for license plate recognition.
"""

import os
import cv2
import numpy as np
import pandas as pd
from data.preprocessing import preprocess_single_car_image, preprocess_single_plate_image
# from models.recognition import decode_predictions


class LicensePlateRecognitionPipeline:
    """
    End-to-end pipeline for license plate recognition.
    """

    def __init__(self, detection_model, recognition_model, idx_to_char, max_length):
        """
        Initialize pipeline with trained models.
        """
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.idx_to_char = idx_to_char
        self.max_length = max_length

        # Configuration
        self.detection_size = (224, 224)  # Input size for detection model
        self.recognition_size = (128, 64)  # Input size for recognition model

    def extract_license_plate(self, img):
        """
        Extract license plate from a car image.
        """
        # Preprocess image for detection
        img_batch, (original_h, original_w) = preprocess_single_car_image(img, self.detection_size)

        # Predict bounding box
        bbox = self.detection_model.predict(img_batch, verbose=0)[0]

        # Convert normalized coordinates back to image coordinates
        xmin = max(0, int(bbox[0] * original_w))
        ymin = max(0, int(bbox[1] * original_h))
        xmax = min(original_w, int(bbox[2] * original_w))
        ymax = min(original_h, int(bbox[3] * original_h))

        # Extract license plate region
        plate_img = img[ymin:ymax, xmin:xmax]

        return plate_img, (xmin, ymin, xmax, ymax)


    def recognize_license_plate(self, plate_img):
        from models.recognition import recognize_characters
        try:
            return recognize_characters(
                plate_img,
                self.recognition_model,
                self.idx_to_char,
                self.max_length
            )
        except ValueError as e:
            print(f"[WARN] Character recognition failed: {e}")
            return "N/A", []


    def process_image(self, img):
        """
        Process a single image through the complete pipeline.
        """
        # Extract license plate
        plate_img, bbox = self.extract_license_plate(img)

        # Recognize text
        license_text, confidences = self.recognize_license_plate(plate_img)

        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Create result image with visualization
        result_img = img.copy()
        xmin, ymin, xmax, ymax = bbox

        # Draw bounding box
        cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw text with background
        text = f"{license_text} ({avg_confidence:.2f})"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

        text_bg_topleft = (xmin, max(0, ymin - text_size[1] - 10))
        text_bg_bottomright = (xmin + text_size[0], ymin)

        # Background rectangle for text
        cv2.rectangle(result_img, text_bg_topleft, text_bg_bottomright, (0, 255, 0), cv2.FILLED)

        # Overlay text
        cv2.putText(result_img, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        return result_img, license_text, bbox, avg_confidence

    
def run_inference_pipeline(pipeline, image_path):
    """
    Run the end-to-end inference pipeline on a single image.

    Args:
        pipeline: Initialized LicensePlateRecognitionPipeline instance
        image_path: Path to the input image

    Returns:
        result_img: Image with license plate and text overlay
        license_text: Recognized license plate text
        confidence: Average confidence score
    """
    import cv2

    # Load image
    img = cv2.imread(image_path)

    # Run pipeline
    result_img, license_text, bbox, confidence = pipeline.process_image(img)

    return result_img, license_text, confidence
