Simple MNIST Neural Network (from scratch, NumPy)

This project implements a simple neural network for handwritten digit classification using the MNIST dataset
. The network is built entirely from scratch using NumPy, without relying on frameworks like TensorFlow or Keras.

The implementation is based on this excellent tutorial
, which served as inspiration and helped me understand the fundamentals of neural networks.

Features

Loading and preprocessing MNIST data

Forward propagation implementation

Cross-entropy loss calculation

Backpropagation for weight updates

Training on the training set and evaluating accuracy on the test set

Requirements

Python 3.10+

NumPy

Pandas (for loading the dataset)

Install dependencies:

pip install numpy pandas

How to run

Clone the repository:

git clone https://github.com/your-username/your-repo.git
cd your-repo


Make sure you have train.csv and test.csv from the Kaggle MNIST competition in the project folder.

Run the script:

python Simple_MNIST_NN_from_scratch.py

Results

After several training epochs, the network should achieve an accuracy of around 90% on the test set.

Acknowledgment

Special thanks to wwsalmon
 for the inspiration and clear explanation of neural network principles.