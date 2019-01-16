import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    train_x = np.asarray(np.random.rand(12, 1), dtype=np.float32) # dtype = np.float32
    train_y = np.asarray(np.random.rand(12, 1), dtype=np.float32) 
    
    return train_x, train_y
