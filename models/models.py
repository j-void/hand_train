import torch
import torch.nn as nn
import os

class Generator(nn.Module):
    
    def __init__(self, noise_dim, pose_dim):
        
        super(Generator, self).__init__()
        
        self.gen = torch.nn.Sequential(
            torch.nn.Linear(in_features=noise_dim, out_features=240, bias=True ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            
            torch.nn.Linear(in_features=240, out_features=240, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            
            torch.nn.Linear(in_features=240, out_features=240, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),

            torch.nn.Linear(in_features=240, out_features=pose_dim, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)
    
class Discriminator(nn.Module):
    
    def __init__(self, pose_dim):
        
        super(Discriminator, self).__init__()
        
        self.disc = torch.nn.Sequential(
            torch.nn.Linear(in_features=pose_dim, out_features=240, bias=True),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=240, out_features=240, bias=True),
            torch.nn.ReLU(),

            torch.nn.Linear(in_features=240, out_features=1, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

def save_model(dict, location, label):
    file_name = os.path.join(location, label+".pth")
    torch.save(dict, file_name)
