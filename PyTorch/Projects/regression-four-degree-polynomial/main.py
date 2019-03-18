import torch
import torch.nn.functional as F

# polynomial of 4th degree 
POLY_DEGREE = 4

w_target = torch.randn(POLY_DEGREE, 1) * 5
print("Target Weights: ", w_target)

b_target = torch.randn(1) * 5
print("Target Bias: ", b_target)


def make_features(x):
    '''
    Builds features i.e. a matrix with columns [x, x**2, x**3, x**4]
    '''
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE)], 1)


def f(x):
    """
    Approximated Function
    """
    return x.mm(w_target) + b_target.item()


def poly_desc(W, b):
    result = 'y = '
    