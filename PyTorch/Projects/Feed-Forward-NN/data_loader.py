import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 100
def load_data():
    train_dataset = torchvision.datasets.MNIST(root='data/', \
            train=True, transform=transforms.ToTensor(), download=True)
    
    test_dataset = torchvision.datasets.MNIST(root='data/', \
            train=False, transform=transforms.ToTensor())
    
    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,\
            batch_size=batch_size, shuffle=True)
        
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,\
            batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
