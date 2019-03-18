import torch
import torch.nn as nn
from torch.autograd import Variable

class LinearRegression(nn.Module):
    """
    @brief: Model Class for Linear Regression
    """
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

# Usage

# Instantiate model class
input_dim = 1
output_dim = 1

model = LinearRegression(input_dim=input_dim, output_dim=output_dim)

# Instantiate loss class
criterion = nn.MSELoss()

# Instantiate optimizer class
learning_rate = 0.001
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=0.9)

# Train the Model
epochs = 100
for epoch in range(epochs):
    pass
