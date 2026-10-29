# Dual-Framework Image Classifier (PyTorch + TensorFlow)

The same convolutional neural network (CNN) trained in both **PyTorch** and **TensorFlow/Keras** on the Fashion-MNIST dataset, compared side by side, with a **FastAPI** endpoint that serves predictions from the PyTorch model.

## Overview

- Built the same CNN architecture in PyTorch and TensorFlow/Keras
- Trained and evaluated both on Fashion-MNIST (10 clothing classes)
- Compared test accuracy and training time
- Served the PyTorch model through a FastAPI image-upload endpoint

## Dataset

Fashion-MNIST: 60,000 training images and 10,000 test images, 28x28 grayscale, 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

## Model architecture

Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dropout(0.3) → Dense(10)

Trained for 8 epochs with the Adam optimizer in both frameworks.

## Results

| Framework  | Test accuracy | Training time |
|------------|---------------|---------------|
| PyTorch    | 0.9102        | 59s           |
| TensorFlow | 0.9140        | 29s           |

Both frameworks reached comparable accuracy. The 0.4% gap is within normal run-to-run variation, so this is not a claim that one framework is better.

Training times come from a single run on a Colab T4 GPU and are not a controlled benchmark. The two pipelines load data differently (PyTorch DataLoader vs in-memory arrays in TensorFlow), which likely explains most of the time difference.

## API

A FastAPI app exposes two endpoints:

- `GET /health` returns `{"status": "ok"}`
- `POST /predict` accepts an image file and returns the predicted class and confidence

Example test result on the first Fashion-MNIST test image:

```
{'prediction': 'Ankle boot', 'confidence': 0.9993}
```

The API was tested inside Google Colab using FastAPI's `TestClient`. It is not hosted on a public server.

## Tech stack

Python, PyTorch, TensorFlow/Keras, FastAPI, NumPy, Pillow

## Repository contents

| File | Description |
|------|-------------|
| `dualnet_fashion_classifier.ipynb` | Full notebook: training, comparison and API test |
| `pytorch_cnn.pt` | Saved PyTorch model weights |
| `tf_cnn.keras` | Saved TensorFlow/Keras model |
| `requirements.txt` | Python dependencies |

## How to run

1. Open `dualnet_fashion_classifier.ipynb` in Google Colab
2. Set the runtime to GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells (Runtime → Run all)

## Limitations

- Fashion-MNIST is a small benchmark dataset; results will not transfer directly to real-world images
- Single training run per framework, no hyperparameter tuning
- No drift monitoring or hosted deployment

## Author

Abu Shahma Khan | [LinkedIn](https://linkedin.com/in/abu-shahma-khan-bb5120295) | [GitHub](https://github.com/abu-shahma12)
