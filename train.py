import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import joblib
import os
from data_prep.renderMPpose import *
from models.models import *

output = joblib.load("data/islrtc_hand_train/lhpts.pkl") 
handpts = output["joints"]
dims = handpts[0].flatten().shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 512
num_steps = len(handpts)

generator = Generator(dims, dims).to(device)
discriminator = Discriminator(dims).to(device)

discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=0.0002, momentum=0.5)
generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.0002, momentum=0.5)

criterion = torch.nn.BCELoss()

if not os.path.exists("tmp"):
    os.makedirs("tmp")

for epoch in range(num_epochs):
    
    for i, real_pose in enumerate(handpts):
        if i == num_steps:
            break
        real_pose_tensor = torch.tensor(real_pose/128, dtype=torch.float)
        real_pose_tensor = real_pose_tensor.flatten().to(device)
        
        batch_size = real_pose_tensor.shape[0]
        # Train Discriminator
        for _ in range(8):
        
            fake_pose_tensor = generator(torch.randn(1, dims).to(device))
            
            real_outputs = discriminator(real_pose_tensor).view(-1)
            lossD_real = criterion(real_outputs, torch.ones_like(real_outputs))
            fake_outputs = discriminator(fake_pose_tensor).view(-1)
            lossD_fake = criterion(fake_outputs, torch.zeros_like(fake_outputs))
            
            lossD = (lossD_real + lossD_fake) / 2
            discriminator.zero_grad()
            lossD.backward(retain_graph=True)

            discriminator_optimizer.step()

        # Train Generator
        outputs = discriminator(fake_pose_tensor)
        lossG = criterion(outputs, torch.ones_like(outputs))
        generator.zero_grad()
        lossG.backward()

        generator_optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] , Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
        with torch.no_grad():
            fake = generator(torch.randn(1, dims).to(device)).reshape(-1, 1, 21, 2)
            fake_pts = fake[0][0].numpy()*128
            fake_img = np.zeros((128, 128, 3), dtype=np.uint8)
            display_single_hand_skleton(fake_img, fake_pts.astype(int))
            cv2.imwrite(f'tmp/out_e{epoch}.png', fake_img)
        

