# MNIST Neural Network from Scratch

This project is a simple implementation of a **neural network built entirely from scratch using NumPy**.  
It classifies handwritten digits from the **MNIST dataset** without relying on any machine learning frameworks like TensorFlow or PyTorch.

> **Inspiration:** The project is based on the excellent [Kaggle notebook by wwsalmon](https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras), which helped me understand the core principles behind how neural networks work.

---

## 📌 Features
- Load and preprocess MNIST data (`train.csv`, `test.csv`)
- **Forward propagation** implementation
- **Cross-entropy loss** calculation
- **Backpropagation** for weight updates
- Training & accuracy evaluation on test data (~90% after several epochs)
---

## 🛠 Requirements
- **Python 3.10+**
- **NumPy**
- **Pandas** (for data loading)