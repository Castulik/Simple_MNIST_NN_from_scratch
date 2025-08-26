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

## BEST PERFORMANCE
- Learning rate: 0.1
- iteration: 0 | accuracy: 0.0654 | loss: 8.3716
- Learning rate: 0.1
- iteration: 50 | accuracy: 0.7078 | loss: 0.9210
- ...
- ...
- ...
- Learning rate: 0.1
- iteration: 1950 | accuracy: 0.9366 | loss: 0.2152
- Learning rate: 0.1
- iteration: 2000 | accuracy: 0.9375 | loss: 0.2129

Acuract test dat: 0.916

## 🛠 Requirements
- **Python 3.10+**
- **NumPy**
- **Pandas** (for data loading)