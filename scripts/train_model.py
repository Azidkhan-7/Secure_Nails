import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["Healthy", "Liver_disease", "Psoriasis_disease"]
IMG_SHAPE = (299, 299)
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 20
RANDOM_SEED = 42

def count_images_in_dir(directory, classes):
    counts = {}
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        if not os.path.exists(cls_path):
            counts[cls] = 0
            continue
        cnt = sum(1 for f in os.listdir(cls_path) if os.path.splitext(f)[1].lower() in valid_exts)
        counts[cls] = cnt
    return counts

def build_aetherdx_model(num_classes=3):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SHAPE[0], IMG_SHAPE[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = Dense(1024, activation='relu', name='dense_1024')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name="AetherDx_InceptionV3")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train(data_dir="dataset", epochs=DEFAULT_EPOCHS, output_path=None):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")
    test_dir = os.path.join(data_dir, "test")

    print("=" * 60)
    print("AetherDx (InceptionV3) Training Pipeline")
    print("=" * 60)

    train_counts = count_images_in_dir(train_dir, CLASS_NAMES)
    val_counts = count_images_in_dir(val_dir, CLASS_NAMES)
    test_counts = count_images_in_dir(test_dir, CLASS_NAMES)

    print("\nDataset Class Counts:")
    for cls in CLASS_NAMES:
        print(f"  {cls:20s} | Train: {train_counts[cls]:4d} | Val: {val_counts[cls]:4d} | Test: {test_counts[cls]:4d}")

    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())

    if total_train == 0:
        print(f"\n[ERROR] No training images found in '{train_dir}'.")
        print("Please populate the dataset directory structure with verified clinical images before training.")
        sys.exit(1)

    print(f"\nTotal: Train={total_train}, Val={total_val}, Test={total_test}")

    # Set seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode='categorical',
        seed=RANDOM_SEED
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode='categorical',
        shuffle=False,
        seed=RANDOM_SEED
    )

    print("\nBuilding model...")
    model = build_aetherdx_model(num_classes=len(CLASS_NAMES))
    print(f"Model compiled. Target Grad-CAM layer 'mixed10' present: {'mixed10' in [l.name for l in model.layers]}")

    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )

    if total_test > 0:
        print("\nEvaluating on test set...")
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SHAPE,
            batch_size=BATCH_SIZE,
            classes=CLASS_NAMES,
            class_mode='categorical',
            shuffle=False
        )
        test_loss, test_acc = model.evaluate(test_generator)
        print(f"Test Accuracy: {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")

        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))

    if output_path is None:
        output_path = os.path.join("aetherdx", "model", "Inception_friend_version.keras")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved successfully to: {output_path}")

    # Also mirror into models/ folder if models folder exists
    models_alt = os.path.join("models", "Inception_friend_version.keras")
    os.makedirs(os.path.dirname(models_alt), exist_ok=True)
    model.save(models_alt)
    print(f"Model mirrored to fallback path: {models_alt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AetherDx InceptionV3 Model")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--output", type=str, default=None, help="Output .keras file path")
    args = parser.parse_args()

    train(data_dir=args.data_dir, epochs=args.epochs, output_path=args.output)
