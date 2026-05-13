# 🐱🐶 Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images as either a **cat** or a **dog**. The dataset is sourced from Kaggle's classic *Dogs vs. Cats* competition and loaded via TensorFlow Datasets.

---

## 📌 Project Overview

This project demonstrates a complete deep learning pipeline for binary image classification:

- Dataset download from Kaggle
- Data loading and preprocessing via TensorFlow Datasets (`tfds`)
- Building a custom CNN from scratch
- Training with validation tracking
- Diagnosing overfitting from accuracy/loss curves
- Running inference on a custom test image

---

## 📁 Project Structure

```
cat_vs_dog_prediction.ipynb   # Main Jupyter Notebook (all code)
kaggle.json                   # Kaggle API credentials (user-provided)
cat.jpg                       # Sample test image for prediction
README.md                     # Project documentation
```

---

## 🛠️ Requirements

### Python Version
- Python 3.8+

### Libraries

Install all dependencies with:

```bash
pip install tensorflow tensorflow-datasets opencv-python matplotlib kaggle
```

| Library | Purpose |
|---|---|
| `tensorflow` | Model building, training, and inference |
| `tensorflow-datasets` | Loading the `cats_vs_dogs` dataset |
| `keras` | High-level neural network API (bundled with TF) |
| `opencv-python` (`cv2`) | Loading and preprocessing the test image |
| `matplotlib` | Plotting accuracy and loss curves |
| `kaggle` | Downloading the dataset via Kaggle CLI |

---

## 🔑 Kaggle API Setup

The dataset is downloaded from Kaggle. You need a Kaggle account and an API key.

1. Go to [https://www.kaggle.com](https://www.kaggle.com) → Account → **Create New API Token**
2. This downloads a `kaggle.json` file
3. Place `kaggle.json` in your working directory (the notebook handles copying it to `~/.kaggle/`)

The notebook runs these setup steps automatically:

```bash
rm -rf ~/.kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c dogs-vs-cats
```

---

## 📊 Dataset

- **Source:** [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) / TensorFlow Datasets `cats_vs_dogs`
- **Split:** 80% training / 20% validation
- **Labels:** `0 = Cat`, `1 = Dog`
- **Preprocessing:**
  - All images resized to `256 × 256`
  - Pixel values normalized to `[0, 1]`
  - Batched in groups of `32`

```python
(data, info) = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)
```

---

## 🧠 Model Architecture

A custom Sequential CNN with 3 convolutional blocks followed by fully connected layers.

```
Input: (256, 256, 3)
    │
    ├── Conv2D(32, 3×3, relu)
    ├── BatchNormalization
    ├── MaxPool2D(2×2, stride=2)
    │
    ├── Conv2D(64, 3×3, relu)
    ├── BatchNormalization
    ├── MaxPool2D(2×2, stride=2)
    │
    ├── Conv2D(128, 3×3, relu)
    ├── BatchNormalization
    ├── MaxPool2D(2×2, stride=2)
    │
    ├── Flatten
    │
    ├── Dense(128, relu)
    ├── Dropout(0.1)
    ├── Dense(64, relu)
    ├── Dropout(0.1)
    │
    └── Dense(1, sigmoid)   → Output: probability [0, 1]
```

### Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## 🚀 Training

```python
history = model.fit(train_data, epochs=10, validation_data=test_data)
```

- **Epochs:** 10
- **Optimizer:** Adam
- **Loss:** Binary Cross-Entropy
- **Batch Size:** 32

---

## 📈 Results & Observations

After training, accuracy and loss curves are plotted for both training and validation sets.

**Key Finding:** The model shows signs of **overfitting** — it performs well on training data but poorly on validation data.

### Planned Improvements to Reduce Overfitting

The notebook identifies the following strategies to address overfitting:

1. **L1/L2 Regularization** — penalize large weights
2. **Dropout Layers** — randomly deactivate neurons during training
3. **Batch Normalization** — stabilize and normalize layer inputs
4. **Data Augmentation** — artificially expand training data with flips, rotations, etc.
5. **Reduce Model Complexity** — simplify the architecture

---

## 🔍 Inference on a Custom Image

The model can predict on any new image. The notebook uses `cat.jpg` as a sample:

```python
import cv2

test_image = cv2.imread('/content/cat.jpg')
test_image = cv2.resize(test_image, (256, 256))
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = test_image / 255.0
test_input = test_image.reshape((1, 256, 256, 3))

prediction = model.predict(test_input)

if prediction > 0.5:
    print("it is a dog")
else:
    print("it is a cat")
```

- **Output > 0.5** → Dog
- **Output ≤ 0.5** → Cat

---

## ▶️ How to Run

1. Clone or download this repository
2. Place your `kaggle.json` in the same directory
3. Open `cat_vs_dog_prediction.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells from top to bottom

> **Tip:** Running on Google Colab with a GPU runtime is recommended for faster training.

---

## 🔮 Future Work

- Implement data augmentation to improve generalization
- Experiment with transfer learning (e.g., MobileNetV2, VGG16, ResNet)
- Add L2 regularization to convolutional and dense layers
- Increase dropout rates
- Deploy the model as a web app using Streamlit or Flask

---

## 📄 License

This project is for educational purposes. Dataset usage is subject to [Kaggle's competition rules](https://www.kaggle.com/c/dogs-vs-cats/rules).
