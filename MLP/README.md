# Multi-Layer Perceptron (MLP)

## 1. Introduction

Multi-Layer Perceptron (MLP) is a type of Artificial Neural Network (ANN). It consists of an input layer, one or more hidden layers, and an output layer.

MLP is a feedforward neural network that learns patterns from data using Forward Propagation and Backpropagation.

## 2. MLP Architecture

![Multi-Layer Perceptron Architecture](images/mlp_architecture.png)

### Example Architecture

The above diagram represents an MLP with:

- Input Layer: 2 Neurons
- Hidden Layer: 2 Neurons
- Output Layer: 2 Neurons

**Architecture:**

\[
2\rightarrow2\rightarrow2
\]

### Layer Description

**Input Layer**
- Receives input features.
- Passes data to the hidden layer.

**Hidden Layer**
- Calculates weighted sums.
- Adds bias.
- Applies activation functions.
- Learns patterns from data.

**Output Layer**
- Generates final predictions.
- Number of neurons depends on the prediction task.

## 3. Components of MLP

- **Neuron:** Processes input information.
- **Weight:** Controls connection strength.
- **Bias:** Adjusts neuron output.
- **Activation Function:** Introduces non-linearity.

## 4. Mathematical Concept

### Weighted Sum

\[
z=\sum_{i=1}^{n}w_ix_i+b
\]

Where:
- x = Input
- w = Weight
- b = Bias
- z = Weighted Sum

### Activation Function

\[
a=f(z)
\]

### Mathematical Example

Given:

\[
x_1=2,\quad x_2=3
\]

\[
w_1=0.5,\quad w_2=0.2
\]

\[
b=0.1
\]

Weighted Sum:

\[
z=(2\times0.5)+(3\times0.2)+0.1
\]

\[
z=1.7
\]

Using ReLU:

\[
f(z)=\max(0,z)
\]

\[
\boxed{f(1.7)=1.7}
\]

## 5. Activation Functions

| Activation | Application |
|---|---|
| ReLU | Hidden Layers |
| Sigmoid | Binary Classification |
| Tanh | Hidden Layers |
| Softmax | Multiclass Classification |
| Linear | Regression |

## 6. Forward Propagation

Forward Propagation passes input data through the network to generate predictions.

### Steps

1. Receive input data.
2. Multiply inputs by weights.
3. Add bias.
4. Apply activation function.
5. Pass output to the next layer.
6. Generate final prediction.

### Mathematical Representation

\[
Z^{(1)}=XW^{(1)}+b^{(1)}
\]

\[
A^{(1)}=f(Z^{(1)})
\]

\[
Z^{(2)}=A^{(1)}W^{(2)}+b^{(2)}
\]

\[
\hat Y=g(Z^{(2)})
\]

## 7. Loss Calculation

Loss measures the difference between actual and predicted outputs.

### Mean Squared Error (MSE)

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

The objective is to minimize the loss.

## 8. Backpropagation

Backpropagation calculates gradients of the loss with respect to weights and biases using the Chain Rule.

### Sigmoid Derivative

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

### Weight Gradient

\[
\frac{\partial L}{\partial w}=\delta x
\]

## 9. Weight Update

Gradient Descent updates weights using:

\[
w_{new}=w_{old}-\eta\frac{\partial L}{\partial w}
\]

Where:
- w = Weight
- η = Learning Rate
- L = Loss Function

## 10. MLP Training Process

1. Input Data
2. Weight Initialization
3. Forward Propagation
4. Loss Calculation
5. Backpropagation
6. Gradient Calculation
7. Optimizer
8. Weight Update
9. Repeat Training

## 11. Optimizers Used in MLP

- Gradient Descent
- SGD
- Momentum
- RMSprop
- Adam
- AdamW
- Adagrad
- Adadelta
- Nadam
- RAdam

## 12. Applications of MLP

- Classification
- Regression
- House Price Prediction
- Fraud Detection
- Disease Prediction
- Credit Risk Analysis
- Customer Churn Prediction
- Pattern Recognition

## 13. Advantages

- Easy to implement.
- Learns nonlinear relationships.
- Supports classification and regression.
- Can learn complex patterns.
- Supports different activation functions and optimizers.

## 14. Limitations

- Can overfit training data.
- Requires feature preprocessing.
- May require large training datasets.
- Does not naturally preserve spatial information in images.
- Large networks can be computationally expensive.

## 15. Python Implementation Using PyTorch

```python
import torch
import torch.nn as nn

# Input data
X = torch.tensor([
    [0.5, 0.8],
    [0.2, 0.4],
    [0.7, 0.3],
    [0.9, 0.6]
])

# Target output
y = torch.tensor([
    [0.05, 0.95],
    [0.10, 0.90],
    [0.80, 0.20],
    [0.90, 0.10]
])

# MLP Model
model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 2),
    nn.Sigmoid()
)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1
)

# Training
for epoch in range(1000):

    prediction = model(X)

    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch: {epoch+1}, "
            f"Loss: {loss.item():.6f}"
        )

print("Final Predictions:")
print(model(X).detach())
```

## 16. Conclusion

Multi-Layer Perceptron is an important feedforward neural network architecture used in Machine Learning and Deep Learning.

It learns patterns through interconnected neurons and improves predictions using Backpropagation and optimization algorithms.

Understanding MLP provides a strong foundation for learning advanced architectures such as CNN, RNN, LSTM, and Transformers.

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Multi-Layer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)

