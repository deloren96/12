from math import factorial as f
import random
'''
from pr import *
roll(37)
'''
def pr(n,k, b):
    #n = 2**n
    return (f(n)/(f(n-k)*f(k)))/(b**n)
    

def br(n,k, p):
    #n = n+1
    k = k+1
    print(n,k, k/n)
    return (f(n)/(f(n-k)*f(k)))*p**k*(1-p)**(n-k)

def per(n, g):
    #print("i",g-n+g, g)
    n = g-n+g
    if n > g:
        return (n-g)/g
    else:
        return (g-n)/g

def roll(j):
    g = 0
    z,r,b = 1,18,18
    for i in range(37, 37+j+1):
    
        print(i,"z",z, per(z/i, 1/37), "z",r, per(r/i,  (1-1/37)/2), "z",b, per(b/i,  (1-1/37)/2),  )
        #print(i,"z",z, (z+1)/i, "z",r, (r+1)/i, "z",b, (b+1)/i,  )
        #print(br(i,z, 1/37), br(i,r, 1/2), br(i,b, 1/2))
        x = random.randint(0, 36)
        if x == 1:
            g = g+1
        if x in [1, 3 ,5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]:
            r = r+1
        elif x == 0:
            z = z+1
        else:
            b = b+1
    print(g)    
br(4+1,1+1, 1/2)

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
print(logits)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")