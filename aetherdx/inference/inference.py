import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

CLASS_NAMES = ["Healthy", "Liver_disease", "Psoriasis_disease"]

def load_aetherdx_model(model_path="aetherdx/model/Inception_friend_version.keras"):
    if not os.path.exists(model_path):
        # fallback to root path if run from scripts
        if os.path.exists(os.path.basename(model_path)):
            model_path = os.path.basename(model_path)
    return tf.keras.models.load_model(model_path)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def run_inference(model, img_path):
    img_array = preprocess_image(img_path)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    
    return {
        "prediction": CLASS_NAMES[class_idx],
        "confidence": float(confidence),
        "probabilities": pred[0].tolist(),
        "model": "AetherDx InceptionV3",
        "preprocessed_image": img_array
    }
