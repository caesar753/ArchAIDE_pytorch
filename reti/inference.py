#importing libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn


from torchvision import models
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import copy
import os

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from datetime import datetime

# infer_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

infer_transform= transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])

# save_path = "C:\\Users\Quirino\Desktop\Reti\Trained_models\\"
save_path = "..\\Data\\Trained_models\\"

to_load = input("Enter the model to load \n")

load_model = (str(save_path) + str(to_load))
model = torch.load(load_model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  model = model.cuda()
#   model.to(device)

model.eval()

while (1):
    to_infer = input("Insert path where is located the image to predict \n")

    image = Image.open(to_infer)
    image = infer_transform(image).unsqueeze(0).cuda()

    output = model(image)
    # print(output)

    #Only prediction
    # prediction = int(torch.max(output.data, 1)[1].cpu().numpy())

    #Prediction with confidence level
    probs = nn.functional.softmax(output, dim=1)
    # print(probs)
    confidence = (torch.max(probs.data, 1))[0].cpu().numpy()
    prediction = (torch.max(probs.data, 1))[1].cpu().numpy()
    print('The prediction is %d with a confidence level of %.2f %%' % (prediction, (100* confidence)))