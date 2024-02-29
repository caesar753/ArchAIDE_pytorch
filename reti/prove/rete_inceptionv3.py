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

#defining function that transforms images for training (cropping, resizing, flipping), and normalize
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])



#defining function that transforms images for testing/validation (resizing), and normalize
test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
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
batch_size = 10

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
def imshow(img, title):
    img = torchvision.utils.make_grid(img, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
#generating an iterator
dataiter = iter(train_loader)

#calling next image and label with iterator
images, labels = dataiter.next()


#function that show images associated to labels
imshow(images, [train_data.classes[i] for i in labels])

#defining dropout
dropout = 0.10
#defining intermediate features
intfeat = 20
#downloading model 
model_down = "inception_v3"

prova = open("C:\\Users\\quirino\\Desktop\\prova.txt", 'a')
prova.write(model_down+ "\n")
prova.close()

model = torch.hub.load('pytorch/vision:v0.10.0', model_down, pretrained=True, force_reload=True)

#loading the model
model

#set aux_logist to False if in transformation function I'm not setting image size >= (299x299), if so I can set it to True
model.aux_logits = False

for parameter in model.parameters():
    parameter.requires_grad = False

# only two linear layers and dropout
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, intfeat),
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
lr = 0.001
optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr)

#setting numbers of epoch
num_epochs = 5

summary_acc_train = []
output_epoch = "" 

#for cycle for training
for epoch in range(num_epochs):
    
    total_batch = len(train_data)//batch_size

    acc_current_train = 0.0

    for i, data in enumerate(train_loader):
        
        #pass data to cuda device
        batch_images, batch_labels = data
        if torch.cuda.is_available():
          batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda()

        X = batch_images
        # print(X)
        Y = batch_labels
        # print(Y)
        # print(i)
      
        #IF YOU ARE *NOT* USING aux_logits
        if (model.aux_logits == False):
          pre = model(X)
          cost = loss(pre, Y)
          _, preds = torch.max(pre, 1)

        
        #ELSE IF YOU ARE USING aux_logits: in these lines of code pre are the outputs of the model applied to the batch, then pre is converted into a tensor called "output"
        #with wich is calculated the loss function
        else:
          output, pre = model(X)
          cost = loss(output,Y)
          _, preds = torch.max(output, 1)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        #the next three lines are useful to print accuracy during training 
        acc_current_train += torch.sum(preds == Y.data)
        epoch_acc = (100 * acc_current_train.float() / len(train_data))
        summary_acc_train.append(epoch_acc)

        if (i+1) % 5 == 0:
            output_epoch = ('Epoch [%d/%d], lter [%d/%d] Loss: %.4f, Accuracy:%.4f %%'
              %(epoch+1, num_epochs, i+1, total_batch, cost.item(), epoch_acc.item()))
            print(output_epoch)
        #if ((epoch+1) == num_epochs) and (((total_batch - i) % 5) == 1):
        

    scheduler.step()
#evaluating the model
    model.eval()

    correct = 0
    total = 0
    
    loss_current_val = 0.0
    acc_current_val = 0.0

    for j, data in enumerate(test_loader):

        #pass data to cuda device
        images, labels = data
        if torch.cuda.is_available():
          images, labels = images.cuda(), labels.cuda()
        
    #   images = images.cuda()
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        
        loss_current_val += val_cost.item() * images.size(0)
        acc_current_val += torch.sum(val_preds == val_labels.data) 
        if (j+1) % 5 == 0:
          # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
          print(f'Epoch: {epoch+1}, batch {batch_size}, learning rate 0.001: %f %%' %(100 * float(correct) / total))

    #Plot
    val_epoch_loss = loss_current_val / len(test_loader)
    summary_loss_val.append(val_epoch_loss)

    val_epoch_acc = acc_current_val.float() / len(test_loader)
    summary_acc_val.append(val_epoch_acc)

    if val_epoch_acc > best_acc:

         best_acc = val_epoch_acc
         best_model_wts = copy.deepcopy(model.state_dict())
         summary_acc_train.append(epoch_acc)

        
    # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
    #print(f'{num_epochs} epochs, batch {batch_size}, learning rate 0.001: %f %%' %(100 * float(correct) / total))