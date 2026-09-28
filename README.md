# Dog, Cat, and Person Image Classification Web App 🐱🐶👤

A deep learning image classification web application built with **PyTorch** and **Gradio**. The app classifies uploaded images into one of three classes (**Cat**, **Dog**, or **Person**) with confidence scores and generates an annotated output image.

---

## 🌟 Features

- **Custom CNN Architecture**: 4-stage convolutional blocks with batch normalization, ReLU activation, and max pooling, feeding into a multi-layer perceptron classifier.
- **Pretrained Weights**: Preloaded PyTorch model checkpoint (`cnn_128_model-100.pth`).
- **Interactive Gradio UI**: Clean, responsive web interface allowing easy drag-and-drop or webcam image uploads.
- **Visual Feedback**: Generates both text predictions with confidence percentages and an annotated output image displaying the classification banner.

---

## 📂 Project Structure

```text
├── app.py                     # Main Gradio application script
├── requirements.txt           # Project dependencies
├── core/
│   ├── __init__.py            # Package export
│   └── predict.py             # ConvNet model architecture & ImageClassifier inference class
├── model/
│   ├── __init__.py
│   └── cnn_128_model-100.pth  # Pre-trained PyTorch model weights
└── .gitignore                 # Git ignore rules
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/johntakal/Dog-Cat-Person-Image-classification-App.git
cd Dog-Cat-Person-Image-classification-App
```

### 2. Create and Activate a Virtual Environment

```bash
# Using conda
conda create -n classifier_env python=3.12 -y
conda activate classifier_env

# Or using venv
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open your browser and navigate to the local URL provided by Gradio (typically **http://127.0.0.1:7860**).

---

## 🧠 Model Architecture

The classifier uses a custom `ConvNet` designed for 128×128 RGB images:
1. **Block 1**: Conv2d (3 → 32), BatchNorm2d, ReLU, MaxPool2d (2×2)
2. **Block 2**: Conv2d (32 → 64), BatchNorm2d, ReLU, MaxPool2d (2×2)
3. **Block 3**: Conv2d (64 → 128), BatchNorm2d, ReLU, MaxPool2d (2×2)
4. **Block 4**: Conv2d (128 → 256), BatchNorm2d, ReLU, MaxPool2d (2×2)
5. **Classifier**: Linear (16,384 → 512) → ReLU → Linear (512 → 128) → ReLU → Linear (128 → 3)

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
