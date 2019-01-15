import torch.nn as nn
import torch, torchvision
from model import NeuralNet
from data_loader import load_data

def hyper_params():
    global input_size, hidden_size, output_size, num_epochs, batch_size, \
            learning_rate
    input_size = 784
    hidden_size = 500
    output_size = 10
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def network():
    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    return model

def loss_opt(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

def train():
    hyper_params()
    num_epochs = 5

    train_loader, test_loader = load_data() 
    model = network()
    criterion, optimizer = loss_opt(model)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # move tensors to configured device
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # backpass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch: {}, Step: [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, i+1, loss.item()))
    return model, test_loader

def test(model, test_loader):
    # no need to compute gradients, in testing phase
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        print('Accuracy: {}%'.format(100 * correct/total))

model, test_loader = train()
test(model, test_loader)
