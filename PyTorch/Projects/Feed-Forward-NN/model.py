import torch.nn as nn
import torch, torchvision

class NeuralNet(nn.Module):
    """
    Neural Net: FC - ReLU - FC
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)       
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
