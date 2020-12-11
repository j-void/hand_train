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
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_data", type=str, default="data/islrtc_hand_train/lhpts.pkl", help='input pkl for training')
parser.add_argument("--save_output", type=str, default="tmp", help='output directory')
parser.add_argument("--checkpoints", type=str, default="checkpoints", help='output directory')

args = parser.parse_args()


output = joblib.load(args.train_data) 
handpts = output["joints"]
dims = handpts[0].flatten().shape[0]

device = "cuda" if torch.cuda.is_available() else "cpu"

num_epochs = 512
num_steps = len(handpts)
batch_size = 32

generator = Generator(dims, dims).to(device)
discriminator = Discriminator(dims).to(device)

discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=0.0002, momentum=0.5)
generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.0002, momentum=0.5)

criterion = torch.nn.BCELoss()

save_output = args.save_output
if not os.path.exists(save_output):
    os.makedirs(save_output)

checkpoints = args.checkpoints 
if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)

print(f'Training for epochs: {num_epochs}, and num_steps: {num_steps}, device: {device}')

for epoch in range(num_epochs):
    
    for i, real_pose in enumerate(handpts):
        if i == num_steps:
            break
        real_pose_tensor = torch.tensor(real_pose/128, dtype=torch.float)
        real_pose_tensor = real_pose_tensor.flatten().to(device)
        
        # Train Discriminator
        for _ in range(8):
        
            fake_pose_tensor = generator(torch.randn(batch_size, dims).to(device))
            
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
        
        if i % 100 == 0:
            print(f"Epoch: [{epoch}/{num_epochs}], iteration: [{i}/{num_steps}], Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
            with open("iter.out", "w") as text_file:
                print(f"Epoch: [{epoch}/{num_epochs}], iteration: [{i}/{num_steps}], Loss D: {lossD:.4f}, loss G: {lossG:.4f}", file=text_file)

        
        if i % 1000 == 0 and i > 0:
            checkpoints_d = {'state_dict': discriminator.state_dict(), 'optimizer': discriminator_optimizer.state_dict()}
            save_model(checkpoints_d, checkpoints, "latest_D")
            checkpoints_g = {'state_dict': generator.state_dict(), 'optimizer': generator_optimizer.state_dict()}
            save_model(checkpoints_g, checkpoints, "latest_G")
            print("saving latest model")

    if epoch % 1 == 0:
        print(f" End of epoch: {epoch}, Loss D: {lossD:.4f}, loss G: {lossG:.4f}")
        with torch.no_grad():
            fake_img_list = []
            fake = generator(torch.randn(batch_size, dims).to(device)).reshape(-1, 1, 21, 2)
            for i in range(len(fake)):
                fake_pts = fake[i][0].numpy()*128
                fake_img = np.zeros((128, 128, 3), dtype=np.uint8)
                display_single_hand_skleton(fake_img, fake_pts.astype(int))
                fake_img_list.append(fake_img)
            vis = np.concatenate(fake_img_list, axis=1)
            cv2.imwrite(f'tmp/fake_e{epoch}.png', vis)
            
    if epoch % 10 == 0 and epoch > 0:
        checkpoints_d = {'state_dict': discriminator.state_dict(), 'optimizer': discriminator_optimizer.state_dict()}
        save_model(checkpoints_d, checkpoints, "epoch_"+str(epoch)+"_D")
        checkpoints_g = {'state_dict': generator.state_dict(), 'optimizer': generator_optimizer.state_dict()}
        save_model(checkpoints_g, checkpoints, "epoch_"+str(epoch)+"_G")
        print("saving model for epoch:"+str(epoch))
        

