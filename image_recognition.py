# Project 4: Image Recognition Using AI 🖼️
# Using Pre-trained MobileNetV2 Model

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

print("=" * 55)
print("   🖼️  AI Image Recognition System")
print("   Model: MobileNetV2 (Pre-trained on ImageNet)")
print("=" * 55)

# Step 1: Load Pre-trained Model
print("\n🔧 Loading pre-trained MobileNetV2 model...")
model = MobileNetV2(weights='imagenet')
print("✅ Model loaded successfully!")
print(f"📊 Model trained on: 1000 ImageNet categories")

# Step 2: Define your local images
# ⚠️ Make sure these image files are in the same folder!
images_info = [
    {"filename": "car.jpg",    "label": "Image 1"},
    {"filename": "flower.jpg", "label": "Image 2"},
]

# Step 3: Recognition Function
def recognize_image(img_path):
    print(f"\n{'='*55}")
    print(f"🔍 Analyzing: {img_path}")
    print(f"{'='*55}")

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    results = decode_predictions(predictions, top=3)[0]

    print(f"\n🎯 TOP 3 PREDICTIONS:")
    print("-" * 40)
    for i, (id, name, confidence) in enumerate(results, 1):
        bar = "█" * int(confidence * 30)
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{emoji} {name.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence*100:.1f}% {bar}")

    top_name = results[0][1].replace('_', ' ').title()
    top_conf = results[0][2] * 100
    print(f"\n✅ Final Answer: {top_name} ({top_conf:.1f}% confident)")
    return results

# Step 4: Run Recognition
print("\n🚀 Starting Image Recognition...")
all_results = {}
found_images = []

for img_info in images_info:
    if os.path.exists(img_info["filename"]):
        results = recognize_image(img_info["filename"])
        all_results[img_info["filename"]] = results
        found_images.append(img_info)
    else:
        print(f"\n⚠️  File not found: {img_info['filename']}")
        print(f"   Please add this image to your project folder!")

# Step 5: Show Visual Results
if found_images:
    print("\n\n📊 Generating visual results...")
    fig, axes = plt.subplots(1, len(found_images),
                             figsize=(5 * len(found_images), 6))

    if len(found_images) == 1:
        axes = [axes]

    for ax, img_info in zip(axes, found_images):
        img = image.load_img(img_info["filename"],
                             target_size=(224, 224))
        results = all_results[img_info["filename"]]
        top_name = results[0][1].replace('_', ' ').title()
        top_conf = results[0][2] * 100

        ax.imshow(img)
        ax.set_title(
            f"🎯 {top_name}\n{top_conf:.1f}% confident",
            fontsize=12,
            fontweight='bold',
            color='darkgreen'
        )
        ax.axis('off')

    plt.suptitle("🖼️ AI Image Recognition Results",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('recognition_results.png',
                dpi=150, bbox_inches='tight')
    print("✅ Chart saved as: recognition_results.png")
    plt.show()

# Step 6: Summary
print("\n" + "=" * 55)
print("📋 RECOGNITION SUMMARY")
print("=" * 55)
for img_info in found_images:
    results = all_results[img_info["filename"]]
    top = results[0]
    name = top[1].replace('_', ' ').title()
    conf = top[2] * 100
    print(f"🖼️  {img_info['filename']:15} → {name:25} ({conf:.1f}%)")

print("\n" + "=" * 55)
print("✅ Project 4 Complete! Image Recognition Done!")
print("=" * 55)