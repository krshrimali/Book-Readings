import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_data

input_size = 1
output_size = 1
num_epochs = 1000
learning_rate = 0.001

def train(train_x, train_y, model, criterion, optimizer):
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(train_x)
        targets = torch.from_numpy(train_y)

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if(epoch % 5 == 0):
            print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))
    
    predicted = model(torch.from_numpy(train_x)).detach().numpy()

    plt.plot(train_x, train_y, 'ro', label = 'Original Data')
    plt.plot(train_x, predicted, label = 'Fitted Line')
    plt.legend()
    plt.show()

def main():
    # load data
    train_x, train_y = load_data()

    # linear regression model
    model = nn.Linear(input_size, output_size)
    
    # loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    train(train_x, train_y, model, criterion, optimizer)

main()
