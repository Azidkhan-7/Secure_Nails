from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("aetherdx/model/Inception_friend_version.keras")

CLASS_NAMES = ["Healthy", "Liver_disease", "Psoriasis_disease"]

# Pick a real training image you KNOW the label of
img_path = r"C:\Users\LOQ\Desktop\Neha Project\Shree Ganesh\Bhuvan dataset\New_dataset\New_dataset\Train\Healthy\WhatsApp Image 2026-07-21 at 11.22.20 PM.jpeg"
img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

pred = model.predict(img_array)
print("Raw scores:", pred)
print("Predicted:", CLASS_NAMES[np.argmax(pred[0])])