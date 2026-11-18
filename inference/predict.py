
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
import keras

from PIL import Image, ImageOps


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMG_SIZE = 224

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "P2_runs"
    / "resnet50_seed1.keras"
)

SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"


# --------------------------------------------------
# Load classes
# --------------------------------------------------

def load_classes():
    """
    Load class names in the exact order used during training.
    """

    if not SPLITS_PATH.is_file():
        raise FileNotFoundError(
            f"Splits file not found: {SPLITS_PATH}"
        )

    with SPLITS_PATH.open(encoding="utf-8") as file:
        splits = json.load(file)

    classes = splits["classes"]

    if not classes:
        raise ValueError("No classes found in splits.json")

    return classes


# --------------------------------------------------
# Load model
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_model():
    """
    Load the trained ResNet50 model.

    The model is loaded only once and reused
    for subsequent predictions.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image_path):
    """
    Apply exactly the same preprocessing used during training.

    Steps:
        1. Correct EXIF orientation.
        2. Convert to RGB.
        3. Resize to 224x224 using bilinear interpolation.
        4. Convert to float32.
        5. Apply ResNet50 preprocessing.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            image = image.resize(
                (IMG_SIZE, IMG_SIZE),
                Image.Resampling.BILINEAR
            )

            image_array = np.asarray(
                image,
                dtype=np.uint8
            )

    except Exception as error:
        raise ValueError(
            f"Could not process image: {image_path}"
        ) from error

    # Add batch dimension: (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array.astype("float32"),
        axis=0
    )

    # ResNet50-specific preprocessing
    image_array = (
        keras.applications.resnet50.preprocess_input(
            image_array
        )
    )

    return image_array


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_image(image_path):
    """
    Predict the grape leaf disease from an image.

    Returns:
        dict containing:
            - predicted_class
            - confidence
            - probabilities
    """

    model = load_model()
    classes = load_classes()

    image = preprocess_image(image_path)

    probabilities = model.predict(
        image,
        verbose=0
    )[0]

    if len(probabilities) != len(classes):
        raise ValueError(
            "Model output size does not match "
            "the number of classes."
        )

    predicted_index = int(np.argmax(probabilities))

    predicted_class = classes[predicted_index]

    confidence = float(
        probabilities[predicted_index]
    )

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability
        in zip(classes, probabilities)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": class_probabilities
    }


# --------------------------------------------------
# Example execution
# --------------------------------------------------

if __name__ == "__main__":

    image_path = PROJECT_ROOT / "test_leaf.jpg"

    result = predict_image(image_path)

    print("Predicted class:", result["predicted_class"])
    print("Confidence:", f"{result['confidence']:.2%}")

    print("\nClass probabilities:")

    for class_name, probability in result["probabilities"].items():
        print(f"{class_name}: {probability:.2%}")
