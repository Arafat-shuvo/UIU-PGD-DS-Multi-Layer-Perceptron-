import torch
import torch.nn as nn

# Input data
X = torch.tensor([
    [2.0, 3.0],
    [1.0, 4.0],
    [3.0, 2.0],
    [4.0, 1.0]
])

# Target output
y = torch.tensor([
    [1.0],
    [0.0],
    [1.0],
    [0.0]
])

# MLP Model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),

    nn.Linear(4, 3),
    nn.ReLU(),

    nn.Linear(3, 1),
    nn.Sigmoid()
)

# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.01
)

# Training
for epoch in range(1000):

    # Forward propagation
    prediction = model(X)

    # Calculate loss
    loss = criterion(prediction, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch+1}, Loss: {loss.item():.4f}"
        )

# Final prediction
print("Predictions:")
print(model(X).detach())