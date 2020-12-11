import torch
import torch.nn as nn


class Generator(nn.Module):
    
    def __init__(self, noise_dim, pose_dim):
        
        super(Generator, self).__init__()
        
        self.gen = torch.nn.Sequential(
            # Fully Connected Layer 1
            torch.nn.Linear(
                in_features=noise_dim,
                out_features=240,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            # Fully Connected Layer 2
            torch.nn.Linear(
                in_features=240,
                out_features=240,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            # Fully Connected Layer 3
            torch.nn.Linear(
                in_features=240,
                out_features=240,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            # Fully Connected Layer 4
            torch.nn.Linear(
                in_features=240,
                out_features=pose_dim,
                bias=True
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.gen(x)
    
class Discriminator(nn.Module):
    
    def __init__(self, pose_dim):
        
        super(Discriminator, self).__init__()
        
        self.disc = torch.nn.Sequential(
            # Fully Connected Layer 1
            torch.nn.Linear(
                in_features=pose_dim,
                out_features=240,
                bias=True
            ),
            torch.nn.ReLU(),
            # Fully Connected Layer 2
            torch.nn.Linear(
                in_features=240,
                out_features=240,
                bias=True
            ),
            torch.nn.ReLU(),
            # Fully Connected Layer 3
            torch.nn.Linear(
                in_features=240,
                out_features=1,
                bias=True
            ),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)
