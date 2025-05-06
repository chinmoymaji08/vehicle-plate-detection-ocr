"""
License plate detection model definition and training.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from data.preprocessing import load_detection_data

def build_detection_model(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Build license plate detection model based on MobileNetV2.

    Args:
        input_shape: Input image shape (height, width, channels)
        weights: Pre-trained weights, 'imagenet' or None

    Returns:
        Compiled Keras model for license plate detection
    """
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Add custom detection head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output layer: [xmin, ymin, xmax, ymax] normalized coordinates
    output = layers.Dense(4, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', iou_metric]
    )

    return model

def detect_license_plate(image, model):
    """
    Detects the license plate bounding box in the input image using the detection model.

    Args:
        image: Input image (as a NumPy array, RGB or BGR)
        model: Trained Keras detection model

    Returns:
        bbox: Tuple of (xmin, ymin, xmax, ymax) in absolute pixel coordinates
    """
    # Resize and normalize image
    input_img = cv2.resize(image, (224, 224))
    input_img = input_img / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # Predict bounding box (normalized)
    pred_bbox = model.predict(input_img)[0]
    h, w = image.shape[:2]

    # Convert to absolute pixel coordinates
    xmin = int(pred_bbox[0] * w)
    ymin = int(pred_bbox[1] * h)
    xmax = int(pred_bbox[2] * w)
    ymax = int(pred_bbox[3] * h)

    # Ensure coordinates are within bounds
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    return (xmin, ymin, xmax, ymax)

def iou_metric(y_true, y_pred):
    """
    Calculate IoU (Intersection over Union) metric.

    Args:
        y_true: Ground truth bounding boxes [xmin, ymin, xmax, ymax]
        y_pred: Predicted bounding boxes [xmin, ymin, xmax, ymax]

    Returns:
        IoU score (tensor)
    """
    y_true_x1, y_true_y1, y_true_x2, y_true_y2 = tf.split(y_true, 4, axis=1)
    y_pred_x1, y_pred_y1, y_pred_x2, y_pred_y2 = tf.split(y_pred, 4, axis=1)

    true_area = (y_true_x2 - y_true_x1) * (y_true_y2 - y_true_y1)
    pred_area = (y_pred_x2 - y_pred_x1) * (y_pred_y2 - y_pred_y1)

    i_x1 = tf.maximum(y_true_x1, y_pred_x1)
    i_y1 = tf.maximum(y_true_y1, y_pred_y1)
    i_x2 = tf.minimum(y_true_x2, y_pred_x2)
    i_y2 = tf.minimum(y_true_y2, y_pred_y2)

    i_width = tf.maximum(0.0, i_x2 - i_x1)
    i_height = tf.maximum(0.0, i_y2 - i_y1)
    intersection_area = i_width * i_height

    union_area = true_area + pred_area - intersection_area

    iou = intersection_area / (union_area + tf.keras.backend.epsilon())
    return tf.reduce_mean(iou)

def load_detection_model(model_path):
    """
    Load a saved detection model with custom metrics.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded model with custom metrics
    """
    custom_objects = {'iou_metric': iou_metric}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def train_detection_model(images, boxes):
    """
    Train the license plate detection model and save it.
    """
    # Define paths
    base_path = r"C:\Users\chinm\License Plate Recognition"
    csv_path = os.path.join(base_path, "data", "Licplatesdetection_train.csv")
    image_dir = os.path.join(base_path, "data", "license_plates_detection_train")
    model_save_path = os.path.join(base_path, "outputs", "detection_model.h5")

    # Load data
    print("Loading detection training data...")
    images, boxes, _ = load_detection_data(csv_path, image_dir)

    # Build model
    print("Building detection model...")
    model = build_detection_model()

    # Train model
    print("Training detection model...")
    model.fit(
        images,
        boxes,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        shuffle=True
    )

    # Save model
    print(f"Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("Detection model training complete.")
    
    return model
