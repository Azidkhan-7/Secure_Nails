import os
import tensorflow as tf

def inspect_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.\n")

    print("Candidate convolutional layers (last 20 layers):")
    for layer in model.layers[-20:]:
        try:
            shape = layer.output.shape
        except:
            shape = "Unknown"
        print(f"- {layer.name} ({layer.__class__.__name__}): output shape {shape}")

    print("\nSelected Grad-CAM layer:")
    print("mixed10")
    print("\nReason:")
    print("For InceptionV3-based models, 'mixed10' is typically the final Concatenate layer "
          "that produces the deepest spatial feature maps (e.g., 8x8x2048) right before "
          "global average pooling and dense classification layers. It captures the highest-level "
          "visual features while maintaining spatial information necessary for generating a heatmap.")

if __name__ == "__main__":
    target = os.path.join("aetherdx", "model", "Inception_friend_version.keras")
    if not os.path.exists(target):
        target = os.path.join("models", "Inception_friend_version.keras")
    inspect_model(target)
