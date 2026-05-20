# 🖼️ Image Recognition Using AI — Deep Learning Project

A pre-trained deep learning image recognition system that identifies
real-world objects from images with high confidence scores.
Built as Project 4 of my AI Engineering journey — DecodeLabs Batch 2026.

---

## 📌 Project Overview

This project implements an AI-powered Image Recognition system using
MobileNetV2 — a pre-trained deep learning model trained on over
1 million real-world images. The system analyzes any input image
and returns the Top 3 predictions with confidence scores.

---

## 🎯 Goal

Implement a basic image recognition task using pre-trained AI
libraries, perform recognition on sample input, and display
the output clearly.

---

## ✅ Key Requirements Completed

- ✔️ Used pre-trained MobileNetV2 model (ImageNet weights)
- ✔️ Performed recognition on real sample images
- ✔️ Displayed Top 3 predictions with confidence scores
- ✔️ Generated visual results chart
- ✔️ Saved output as recognition_results.png

---

## 🧠 Key Skills Learned

- ✅ Using AI libraries (TensorFlow & Keras)
- ✅ Loading and using pre-trained models
- ✅ Image preprocessing for deep learning
- ✅ Understanding model outputs & predictions
- ✅ Decoding ImageNet predictions
- ✅ Visualizing AI results with Matplotlib

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| MobileNetV2 | Pre-trained recognition model |
| ImageNet | 1000-class training dataset |
| NumPy | Array & data processing |
| Matplotlib | Visual results chart |
| Pillow | Image loading & processing |
| VS Code | Development environment |

---

## 📂 Project Structure
---

## ▶️ How to Run

1. Make sure Python is installed
2. Install required libraries:
```bash
pip install tensorflow numpy pillow matplotlib
```
3. Clone this repository:
```bash
git clone https://github.com/farheenrafiq20056-svg/task-4-farheenrafiq.git
```
4. Navigate to project folder:
```bash
cd task-4-farheenrafiq
```
5. Add your own images to the folder (car.jpg, flower.jpg)
6. Run the recognition system:
```bash
python image_recognition.py
```

---

## 💬 Sample Output

=======================================================
🖼️  AI Image Recognition System
Model: MobileNetV2 (Pre-trained on ImageNet)
🔧 Loading pre-trained MobileNetV2 model...
✅ Model loaded successfully!
📊 Model trained on: 1000 ImageNet categories
🚀 Starting Image Recognition...
=======================================================
🔍 Analyzing: car.jpg
🎯 TOP 3 PREDICTIONS:
🥇 Sports Car
Confidence: 94.2% ████████████████████████████
🥈 Racer
Confidence: 3.1%  █
🥉 Convertible
Confidence: 1.2%
✅ Final Answer: Sports Car (94.2% confident)
=======================================================
🔍 Analyzing: flower.jpg
🎯 TOP 3 PREDICTIONS:
🥇 Daisy
Confidence: 97.5% █████████████████████████████
🥈 Sunflower
Confidence: 1.8%
🥉 Vase
Confidence: 0.4%
✅ Final Answer: Daisy (97.5% confident)
=======================================================
📋 RECOGNITION SUMMARY
🖼️  car.jpg        → Sports Car              (94.2%)
🖼️  flower.jpg     → Daisy                   (97.5%)
=======================================================
