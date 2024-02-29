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

import early_stop_train_val

import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime


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
#dropout = 0.10
dropout = input("Please enter a dropout value (float from 0 to 1):\n")
dropout = float(dropout)

#defining intermediate features
#intfeat = 20
intfeat = input("Please enter an intermediate feature value (integer):\n")
intfeat = int(intfeat)

#Defining tolerance and delta for early stop
tol = input("Enter a number of epochs of patience for early stop")
delta = input("Enter a delta between train loss and validation loss")
tol = int(tol)
delta = float(delta)
early_stopper_train_Val = early_stop_train_val.EarlyStopping_Train_Val(tolerance=tol, min_delta=delta)

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

print(f'Summary:\n \
        Neural Networl:{model_down},\n \
        epochs:{num_epochs},\n \
        batch:{batch_size},\n \
        learning rate: {lr},\n \
        Intermediate features: {intfeat},\n \
        Dropout:{dropout},\n \
        optimizer: {optimizer} \n')


now = datetime.now()
now = now.strftime("%Y%m%d%H%M")
save_path = ("C:\\Users\Quirino\Desktop\Reti\Trained_models\\" +\
    str(now) + "_" + str(model_down) + "_" +\
    str(num_epochs) + "epochs_" +\
    str(batch_size) + "batch_"+\
    str(lr) + "LR_" +\
    str(dropout) + "dropout_" +\
    optim_choose + "optimizer_" +\
    ".pth")

print(f'Path where the model is saved is: {save_path}')
# open(save_path, 'a')

#for cycle for training
for epoch in range(num_epochs):
    
    total_batch = len(train_data)//batch_size

    acc_current_train = 0.0
    cost_current_train = 0.0


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
        epoch_acc = acc_current_train.float() / len(train_data)
        cost_current_train += cost.item() * batch_images.size(0)
        epoch_loss = cost_current_train / len(train_data)
        #summary_acc_train.append(epoch_acc)

        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f, Accuracy:%.4f %%'
                 %(epoch+1, num_epochs, i+1, total_batch, cost.item(), (100 * epoch_acc.item())))
          
    scheduler.step()
#evaluating the model
    model.eval()

    correct = 0
    total = 0
    
    loss_current_val = 0.0
    acc_current_val = 0.0
    
    total_batch_val = len(test_data)//batch_size

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
        
        val_cost = loss(outputs, labels)
        
        loss_current_val += val_cost.item() * images.size(0)
        val_epoch_loss = loss_current_val / len(test_data)
        acc_current_val += torch.sum(predicted == labels.data)
        val_epoch_acc = acc_current_val.float() / len(test_data)

        if (j+1) % 5 == 0:
          # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
          print(f'Epoch: {epoch+1}, batch {batch_size}, learning rate {lr}, Loss:{val_cost.item()}, iter: [{j+1}/{total_batch_val}]: %f %%' %(100 * float(correct) / total))
    
    #Plot

    summary_loss_train.append(epoch_loss)
    summary_acc_train.append(epoch_acc)  
    summary_loss_val.append(val_epoch_loss)
    summary_acc_val.append(val_epoch_acc)


    if val_epoch_acc > best_acc:
         best_acc = val_epoch_acc
         best_model_wts = copy.deepcopy(model.state_dict())
         #summary_acc_train.append(epoch_acc)
    
    early_stopper_train_Val(epoch_loss, val_epoch_loss)
    if early_stopper_train_Val.early_stop:
      print("We are at epoch:", epoch+1)
      break
    else:
        print(f"We have not yet achieved early stop value,\
            train_loss is {epoch_loss}, \
            val_loss is {val_epoch_loss}, \
            so their difference is {val_epoch_loss - epoch_loss}")

        
    # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
    #print(f'{num_epochs} epochs, batch {batch_size}, learning rate 0.001: %f %%' %(100 * float(correct) / total))

print(f'Summary: Neural Network:{model_down},\n \
    epochs:{num_epochs},batch:{batch_size}, learning rate: {lr}, Intermediate features: {intfeat}, Dropout:{dropout},\n \
    optimizer: {optimizer} \n \
    Training: Total batch: [{total_batch}], Loss:{epoch_loss}, Accuracy: {100 * epoch_acc}%, \n \
    Validation: batch size:[{total_batch_val}], Loss: {val_epoch_loss}, Accuracy: {100 * val_epoch_acc}%')

print(f'Summary loss train: {summary_loss_train}')
print(f'Summary loss train shape: {np.shape(summary_loss_train)}')
print(f'Summary loss val: {summary_loss_train}')
print(f'Summary loss val shape: {np.shape(summary_loss_val)}')
# summary_acc_train_cpu = summary_acc_train.cpu()
#print(f'Summary acc train shape: {np.shape(summary_acc_train.cpu())}')

sommario_acc_train_array = []
for idx in range(len(summary_acc_train)):
    sommario_acc_train_array.append(summary_acc_train[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_train_array[idx])
print(f'Sommario acc train array shape: {np.shape(sommario_acc_train_array)}')


sommario_acc_val_array = []
for idx in range(len(summary_acc_val)):
    sommario_acc_val_array.append(summary_acc_val[idx].cpu().clone().detach().numpy())
    print(sommario_acc_val_array[idx])
print(f'Sommario acc val array shape: {np.shape(sommario_acc_val_array)}')


#Saving the model
torch.save(model, save_path)


#Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,6))
fig.suptitle(f'Model:{model_down}, Intermediate features: {intfeat}, Dropout:{dropout} Epochs: {num_epochs}, optimizer: {optim_choose} with momentum:{momentum} and gamma: {gamma}, \n Loss function:{loss}, learning rate:{lr}')

x = [i for i in range(num_epochs)]
#print(x)
ax1.set_title("Loss")
ax1.plot(x, summary_loss_train,  label = 'Training Loss')
ax1.plot(x, summary_loss_val, label = 'Validation Loss')
ax1.legend()

sommario_acc_train_array = []
for idx in range(len(summary_acc_train)):
    sommario_acc_train_array.append(summary_acc_train[idx].cpu().clone().detach().numpy())
print(f'Sommario acc train array: {np.shape(sommario_acc_train_array)}')

sommario_acc_val_array = []
for idx in range(len(summary_acc_val)):
    sommario_acc_val_array.append(summary_acc_val[idx].cpu().clone().detach().numpy())
print(f'Sommario acc val array: {np.shape(sommario_acc_val_array)}')

ax2.set_title("Accuracy")
ax2.plot(x, sommario_acc_train_array, label='Training Accuracy')
ax2.plot(x, sommario_acc_val_array, label='Validation Accuracy')
ax2.legend()

plt.show()