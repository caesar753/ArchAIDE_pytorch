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

import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#defining device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device
print(device)

#choosing model 
model_down = input("Which model?")

#inception_v3 needs  an immage 300x300, other neural networks 225x225
if model_down == "inception_v3":
    im_dim = int(299)
else:
    im_dim = int(224)

#defining function that transforms images for training (cropping, resizing, flipping), and normalize
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(im_dim),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])

#defining function that transforms images for testing/validation (resizing), and normalize
test_transform = transforms.Compose([
    transforms.Resize((im_dim, im_dim)),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])

#Defining path
os.chdir("F:\ArchAIDE_nn\Archapp_pytorch\AllClasses_giusto")
PATH = os.getcwd()
#PATH = "F:\ArchAIDE_nn\archapp_pytorch\ "

print(PATH)

TRAIN_FOLDER = os.path.join(PATH, 'Train') 
print(TRAIN_FOLDER)

VAL_FOLDER = os.path.join(PATH, 'Val') 
print(VAL_FOLDER)

#creating dataset for training, using the directory of my Drive, MUST CHANGE for another Drive
#QUIRINO'S PATH
train_data = dsets.ImageFolder(TRAIN_FOLDER, train_transform)
#DUBBINI'S PATH
# train_data = dsets.ImageFolder('/content/drive/MyDrive/Theses/Saraceni/mio_new/MTL', train_transform)

#creating dataset for validation, using the directory of my Drive, MUST CHANGE for another Drive
#QUIRINO'S PATH
test_data = dsets.ImageFolder(VAL_FOLDER, test_transform)
#DUBBINI'S PATH
#test_data = dsets.ImageFolder('/content/drive/MyDrive/Theses/Saraceni/mio_new/MTL', test_transform)

#batch size for training and testing
#batch_size = 10
batch_size = input("Please enter a batch size (integer):\n")
batch_size = int(batch_size)

#dataloader for training, drop_last=True necessary because, if the dataset is not dividable for the batch size and remains only one image training fails
train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

#dataloader for testing
test_loader = DataLoader(test_data, 
                         batch_size=batch_size,
                         shuffle=True)

dataset_len = len(train_data)
print('dataset:', train_data)
print('dataset size:', dataset_len)
class_names = train_data.classes
print('class_names:', class_names)
number_of_classes = len(class_names)
print('class_names length:', number_of_classes)

#function which overlay the image with a grid and transforms it with numpy
def imshow(img, title):[
    img = torchvision.utils.make_grid(img, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()]
    
#generating an iterator
dataiter = iter(train_loader)

#calling next image and label with iterator
images, labels = dataiter.next()


#function that show images associated to labels
imshow(images, [train_data.classes[i] for i in labels])


#defining dropout
#dropout = 0.10
dropout = input("Please enter a dropout value (float from 0 to 1):\n")
dropout = float(dropout)

#defining intermediate features
#intfeat = 20
intfeat = input("Please enter an intermediate feature value (integer):\n")
intfeat = int(intfeat)

#downloading the model 
model = torch.hub.load('pytorch/vision:v0.10.0', model_down, pretrained=True, force_reload=True)

#loading the model
model

#set aux_logist to False if in transformation function I'm not setting image size >= (299x299), if so I can set it to True
model.aux_logits = False

for parameter in model.parameters():
    parameter.requires_grad = False

# only two linear layers and dropout
model.fc = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(model.fc.in_features, intfeat),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(intfeat, number_of_classes)
)

#if you want to use cuda uncomment this line
if torch.cuda.is_available():
  model = model.cuda()
  model = model.to(device)
print(model.fc)

#setting loss function, for now CrossEntropy
loss = nn.CrossEntropyLoss()

#setting optimizer with learning rate, for now RMSProp with lr=0.001
#lr = 0.005
lr = input("Please enter a learning rate (float):\n")
lr = float(lr)
step = input("How many steps (integer)?")
step = int(step)
momentum = input("Please enter a momentum (float):\n")
momentum = float(momentum)
gamma = input("Please enter gamma (float):\n")
gamma = float(gamma)


optim_choose = input("Which optimizer (sgd, rmsprop, adam)?")

if optim_choose == "sgd":
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)


elif optim_choose == "rmsprop":
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

elif optim_choose == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

else:
    print("No valid optimizer")
    exit(0)

scheduler = lr_scheduler.StepLR(optimizer, step_size = step, gamma = gamma)
#setting numbers of epoch
#num_epochs = 10
num_epochs = input("Please enter the number of epochs (integer):\n")
num_epochs = int(num_epochs) 

summary_loss_train = []  
summary_acc_train = []   

summary_loss_val = []    
summary_acc_val = []

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0