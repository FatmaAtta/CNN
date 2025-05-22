# Customizable CNN for MNIST Digit Classification

This repository contains a fully customizable Convolutional Neural Network (CNN) implementation using TensorFlow/Keras to classify handwritten digits from the MNIST dataset. The model architecture, activation functions, optimizer, learning rate, number of layers, and other hyperparameters are fully configurable.

---

## Features

- Supports **1 to 3 convolutional layers** with configurable filters and activations  
- Customizable **fully connected (dense) layers** with adjustable size and count  
- Flexible **dropout placement** and rate for regularization  
- Choice of **activation functions**: ReLU, Sigmoid, Tanh, Softmax  
- Supports multiple optimizers: SGD, Adam, RMSprop  
- Training with **early stopping** and validation support  
- Detailed training and testing time measurement per epoch  
- Easily extendable for other datasets and tasks  

---

## Dataset

Uses the popular [MNIST dataset](http://yann.lecun.com/exdb/mnist/):

- 60,000 training images (28x28 grayscale)  
- 10,000 test images  

---

## Getting Started

### Requirements

- Python 3.8+  
- TensorFlow 2.x  
- NumPy  
- Matplotlib (optional, for visualization)  

```bash
pip install tensorflow numpy matplotlib

CNN(
    activation_fun=ActivationFunction.ReLU,
    conv_layers=3,
    epoch=10,
    optimizer=Optimizer.SGD,
    learning_rate=0.01,
    batch_size=32,
    fc_layers=2,
    fc_size=128,
    dropout_pos=-1,      # Position of dropout layer (-1 means no dropout)
    dropout_rate=0.0
)
```
---

## Parameters Explained

- **activation_fun**  
  Activation function for all layers (e.g., ReLU, Sigmoid, etc.)

- **conv_layers**  
  Number of convolutional layers (maximum 3)

- **epoch**  
  Number of training epochs

- **optimizer**  
  Optimizer to use (SGD, Adam, RMSprop)

- **learning_rate**  
  Learning rate for the optimizer

- **batch_size**  
  Number of samples per training batch

- **fc_layers**  
  Number of fully connected (dense) layers (maximum 4)

- **fc_size**  
  Number of neurons in each dense layer

- **dropout_pos**  
  Index of the dense layer after which to apply dropout (-1 to skip)

- **dropout_rate**  
  Dropout rate (from 0.0 to 1.0)
