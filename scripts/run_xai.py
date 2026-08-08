import os
import sys
import argparse
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.inference import load_aetherdx_model, run_inference
from backend.gradcam import generate_gradcam, save_and_display_gradcam

def main():
    parser = argparse.ArgumentParser(description="Run AetherDx Model with Grad-CAM")
    parser.add_argument("image_path", type=str, help="Path to the image to diagnose")
    args = parser.parse_args()

    image_path = args.image_path
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    print("Loading model...")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Inception_friend_version.keras")
    model = load_aetherdx_model(model_path)
    
    print("Running inference...")
    result = run_inference(model, image_path)
    
    print("\nPrediction:")
    print(result["prediction"])
    print(f"\nConfidence:\n{result['confidence'] * 100:.2f}%")
    
    print("\nGrad-CAM Layer:")
    print("mixed10")
    
    print("\nGenerating Grad-CAM heatmap...")
    heatmap, _ = generate_gradcam(model, result["preprocessed_image"], layer_name="mixed10")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "xai")
    os.makedirs(output_dir, exist_ok=True)
    
    cam_path = os.path.join(output_dir, "gradcam_overlay.jpg")
    
    save_and_display_gradcam(image_path, heatmap, cam_path=cam_path, alpha=0.4)
    
    # Rename for expected names
    if os.path.exists(os.path.join(output_dir, "gradcam_overlay_original.jpg")):
        os.rename(os.path.join(output_dir, "gradcam_overlay_original.jpg"), os.path.join(output_dir, "original.jpg"))
    if os.path.exists(os.path.join(output_dir, "gradcam_overlay_heatmap.jpg")):
        os.rename(os.path.join(output_dir, "gradcam_overlay_heatmap.jpg"), os.path.join(output_dir, "heatmap.jpg"))
        
    print("\nOutputs saved to:")
    print(os.path.join(output_dir, "original.jpg"))
    print(os.path.join(output_dir, "heatmap.jpg"))
    print(os.path.join(output_dir, "gradcam_overlay.jpg"))

if __name__ == "__main__":
    main()
