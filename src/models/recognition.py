"""
Character recognition model and utilities.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint


def create_recognition_model(input_shape=(64, 128, 1), num_classes=37, max_length=10):
    """
    Create a character recognition model that predicts a sequence of characters.
    
    Args:
        input_shape: Shape of input image (H, W, C)
        num_classes: Number of possible characters (including padding)
        max_length: Maximum number of characters to predict
    
    Returns:
        model: Keras model
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Repeat vector to predict max_length characters
    x = layers.RepeatVector(int(max_length))(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))(x)
    
    model = models.Model(inputs=inputs, outputs=x)
    return model


def create_char_mappings():
    """
    Create character-index mappings.
    
    Returns:
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
    """
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}  # start from 1 (0 = padding)
    idx_to_char = {idx + 1: char for idx, char in enumerate(characters)}
    char_to_idx["<PAD>"] = 0
    idx_to_char[0] = "<PAD>"
    
    return char_to_idx, idx_to_char


def recognize_characters(plate_img, model, idx_to_char, max_length):
    """
    Recognize characters from a license plate image using the recognition model.
    
    Args:
        plate_img: Input license plate image
        model: Trained recognition model
        idx_to_char: Dictionary mapping indices to characters
        max_length: Maximum number of characters
    
    Returns:
        predicted_text: Recognized license plate string
        confidences: List of confidence scores
    """
    from data.preprocessing import preprocess_single_plate_image  # moved import here to avoid circular import

    img_batch = preprocess_single_plate_image(plate_img)
    predictions = model.predict(img_batch, verbose=0)
    
    predicted_text = ""
    confidences = []
    
    for i in range(max_length):
        pred = predictions[0][i]
        pred_idx = tf.argmax(pred).numpy()
        confidence = float(pred[pred_idx])
        
        if pred_idx != 0:  # skip padding
            predicted_text += idx_to_char[pred_idx]
            confidences.append(confidence)
    
    return predicted_text, confidences

def train_recognition_model(train_images, train_labels, max_length, char_to_idx, epochs, save_path):
    """
    Train the character recognition model and save the full model.

    Args:
        train_images: Array of input images
        train_labels: Array of label sequences
        max_length: Maximum label length
        char_to_idx: Dictionary mapping characters to indices
        epochs: Number of training epochs
        save_path: Path to save best full model
    Returns:
        model: Trained Keras model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = create_recognition_model(
        input_shape=train_images.shape[1:],
        num_classes=len(char_to_idx),
        max_length=max_length
    )

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        filepath=save_path,
        save_best_only=True,
        save_weights_only=False,  # Save full model, not just weights
        monitor='val_loss',
        mode='min'
    )

    model.fit(
        train_images,
        train_labels,
        batch_size=32,
        epochs=epochs,
        validation_split=0.1,
        callbacks=[checkpoint]
    )

    return model


