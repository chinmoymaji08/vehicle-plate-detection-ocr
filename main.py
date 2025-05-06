import os
import sys

# Add source directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data.preprocessing import load_detection_data, load_recognition_data
from models.detection import train_detection_model
from models.recognition import train_recognition_model
from inference.pipeline import run_inference_pipeline, LicensePlateRecognitionPipeline
from keras.models import load_model

# Paths for your environment
BASE_PATH = r"C:\Users\chinm\License Plate Recognition"
DETECTION_IMAGES = os.path.join(BASE_PATH, "data", "license_plates_detection_train")
RECOGNITION_IMAGES = os.path.join(BASE_PATH, "data", "license_plates_recognition_train")
TEST_IMAGES = os.path.join(BASE_PATH, "data", "test")
CSV_DIR = os.path.join(BASE_PATH, "data")
OUTPUT_DIR = os.path.join(BASE_PATH, "outputs")

# Ensure output directories exist
def create_dirs():
    dirs = [
        os.path.join(OUTPUT_DIR, "logs"),
        os.path.join(OUTPUT_DIR, "predictions")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def main():
    create_dirs()

    # Step 1: Load data
    detection_csv = os.path.join(CSV_DIR, "Licplatesdetection_train.csv")
    recognition_csv = os.path.join(CSV_DIR, "Licplatesrecognition_train.csv")

    detection_images, detection_boxes, _ = load_detection_data(detection_csv, DETECTION_IMAGES)
    recognition_images, recognition_labels, max_length, char_to_idx, idx_to_char = load_recognition_data(recognition_csv, RECOGNITION_IMAGES)

    # Step 2: Train models
    detection_model_path = os.path.join(OUTPUT_DIR, "detection_model.keras")
    recognition_model_path = os.path.join(OUTPUT_DIR, "recognition_model.keras")

    # Train and save detection model
    detection_model = train_detection_model(detection_images, detection_boxes)
    detection_model.save(detection_model_path)

    # Train and save recognition model
    recognition_model = train_recognition_model(
        recognition_images,
        recognition_labels,
        max_length,
        char_to_idx,
        epochs=50,
        save_path=recognition_model_path  # full model save
    )

    # Step 3: Load models
    detection_model = load_model(detection_model_path, compile=False)
    recognition_model = load_model(recognition_model_path, compile=False)

    # Step 4: Initialize pipeline
    pipeline = LicensePlateRecognitionPipeline(
        detection_model=detection_model,
        recognition_model=recognition_model,
        idx_to_char=idx_to_char,
        max_length=max_length
    )

    # Step 5: Run inference
    for fname in os.listdir(TEST_IMAGES):
        if fname.lower().endswith(".jpg"):
            image_path = os.path.join(TEST_IMAGES, fname)
            result_img, license_text, confidence = run_inference_pipeline(pipeline, image_path)
            print(f"{fname}: {license_text} (Confidence: {confidence:.2f})")


if __name__ == "__main__":
    main()


