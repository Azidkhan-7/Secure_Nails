import os
from PIL import Image
from backend.inference import load_aetherdx_model, run_inference
from backend.gradcam import generate_gradcam, save_and_display_gradcam

model = load_aetherdx_model("Inception_friend_version.keras")
img_path = r"C:\Users\LOQ\Desktop\Neha Project\Shree Ganesh\Bhuvan dataset\New_dataset\New_dataset\Train\Healthy\WhatsApp Image 2026-07-21 at 11.22.20 PM.jpeg"
temp_image_path = "outputs/temp_upload.jpg"
image = Image.open(img_path).convert("RGB")
image.save(temp_image_path)

result = run_inference(model, temp_image_path)
heatmap, _ = generate_gradcam(model, result["preprocessed_image"], layer_name="mixed10")
save_and_display_gradcam(temp_image_path, heatmap, cam_path="outputs/xai/test_gradcam_overlay.jpg", alpha=0.4)
print("Success")
