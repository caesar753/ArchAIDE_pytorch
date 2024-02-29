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

import early_stop_val_class as early_stop

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

k = input("Insert the top score for results \n")
k = int(k)
num_epochs = input("Please enter the number of epochs (integer):\n")
num_epochs = int(num_epochs) 

pat = input("Insert how many epochs for patience")
pat = int(pat)
delta = input("insert a delta")
delta = float(delta)
early_stopper = early_stop.EarlyStopper(patience=pat, min_delta=delta)


summary_loss_train = []  
summary_acc_train = []   
summary_acc_train_top = []

summary_loss_val = []    
summary_acc_val = []
summary_acc_val_top = []

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

#for cycle for training
for epoch in range(num_epochs):
    
    total_batch = len(train_data)//batch_size

    acc_current_train = 0.0
    acc_current_train_top = 0.0
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
          #top5 accuracy
          _, preds = torch.max(pre, 1)
          _, predstop = torch.topk(pre, k)
          #transpose the topk matrix because topk vector is a column vector and ground truth vector is a row vector
          predstop = torch.t(predstop)
        #   if (Y.data in preds):
        #     print(f'Predicted max are {preds} and ground truth is {Y.data} \n')
        #   if (Y.data in predstop):
        #     print(f'Predicted top{k} are {predstop} and ground truth is {Y.data} \n')
        
        #ELSE IF YOU ARE USING aux_logits: in these lines of code pre are the outputs of the model applied to the batch, then pre is converted into a tensor called "output"
        #with wich is calculated the loss function
        else:
          output, pre = model(X)
          cost = loss(output,Y)
          #top5 accuracy
          _, preds = torch.max(output, 1)
          _, predstop = torch.topk(output, k)
          #transpose the topk matrix because topk vector is a column vector and ground truth vector is a row vector
          predstop = torch.t(predstop)
        #   if (Y.data in preds):
        #     print(f'Predicted max are {preds} and ground truth is {Y} \n')
        #   if (Y.data in predstop):
        #     print(f'Predicted top{k} are {predstop} and ground truth is {Y} \n')

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        #the next three lines are useful to print accuracy during training 
        acc_current_train += torch.sum(preds == Y.data)
        # if (Y.data in preds):
        #     print(acc_current_train)
        epoch_acc = acc_current_train.float() / len(train_data)
        
        acc_current_train_top += torch.sum(predstop == Y.data)
        # if (Y.data in predstop):
        #     print(acc_current_train_top)
        epoch_acc_top = acc_current_train_top.float() / len(train_data)
        
        cost_current_train += cost.item() * batch_images.size(0)
        epoch_loss = cost_current_train / len(train_data)
        #summary_acc_train.append(epoch_acc)

        if (i+1) % 5 == 0:
            print('Epoch [%d/%d], lter [%d/%d] Loss: %.4f, Accuracy_top1 : %.4f %%, Accuracy_top%d : %.4f %%'
                 %(epoch+1, num_epochs, i+1, total_batch, cost.item(), \
                  (100 * epoch_acc.item()), k, (100 * epoch_acc_top.item())))
          
    scheduler.step()
#evaluating the model
    model.eval()

    correct = 0
    correcttop = 0
    total = 0
    
    loss_current_val = 0.0
    acc_current_val = 0.0
    acc_current_val_top = 0.0
    
    total_batch_val = len(test_data)//batch_size

    for j, data in enumerate(test_loader):

        #pass data to cuda device
        images, labels = data
        if torch.cuda.is_available():
          images, labels = images.cuda(), labels.cuda()
        
    #   images = images.cuda()
        outputs = model(images)
        
        #top5 accuracy
        _, predicted = torch.max(outputs.data, 1)
        _, predictedtop = torch.topk(outputs.data, k)
        predictedtop = torch.t(predictedtop)
        
        # if (labels.data in predicted):
        #     print(f'Predicted max is {predicted} and ground truth is {labels} \n')
        
        # if (labels.data in predictedtop):
        #     print(f'Top{k} predicted is: {predictedtop} and ground truth is {labels} \n')
        
        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()
        correcttop += torch.sum(predictedtop == labels)
        
        # print(f'Correct max is {predicted} and ground truth is {labels} \n')
        # print(f'Correct top5 is {predictedtop} and ground truth is {labels} \n')
        
        val_cost = loss(outputs, labels)
        
        loss_current_val += val_cost.item() * images.size(0)
        val_epoch_loss = loss_current_val / len(test_data)
        
        acc_current_val += torch.sum(predicted == labels.data)
        val_epoch_acc = acc_current_val.float() / len(test_data)

        acc_current_val_top += torch.sum(predictedtop == labels.data)
        val_epoch_acc_top = acc_current_val_top.float() / len(test_data)

        if (j+1) % 5 == 0:
          # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
          print(f'Epoch: {epoch+1}, batch {batch_size}, learning rate {lr}, iter: [{j+1}/{total_batch_val}]: Accuracy: top1: %f %%, top{k}: %f %%' \
            %((100 * float(correct) / total), (100 * float(correcttop / total))))

    #Plot

    summary_loss_train.append(epoch_loss)
    summary_acc_train.append(epoch_acc)
    summary_acc_train_top.append(epoch_acc_top)
    
    summary_loss_val.append(val_epoch_loss)
    summary_acc_val.append(val_epoch_acc)
    summary_acc_val_top.append(val_epoch_acc_top)


    if val_epoch_acc > best_acc:
         best_acc = val_epoch_acc
         best_model_wts = copy.deepcopy(model.state_dict())
         #summary_acc_train.append(epoch_acc)


    if early_stopper.early_stop(val_epoch_loss):
        print("We are at epoch:", epoch+1)
        break   
    else:
        print(f"We have not yet achieved early stop value,\
            min_val_loss is {early_stopper.min_validation_loss}, \
            val_loss is {val_epoch_loss}")
        

    # print('Accuracy of test images with %f epochs and %f: %f %%' % (100 * float(correct) / total))
    #print(f'{num_epochs} epochs, batch {batch_size}, learning rate 0.001: %f %%' %(100 * float(correct) / total))

print(f'Summary: Neural Network:{model_down},\n \
    epochs:{num_epochs},batch:{batch_size}, learning rate: {lr}, Intermediate features: {intfeat}, Dropout:{dropout},\n \
    optimizer: {optimizer} \n \
    Training: Total batch: [{total_batch}], Loss:{epoch_loss}, Accuracy: top1: {100 * epoch_acc}%, top{k}: {100 * epoch_acc_top}% \n \
    Validation: batch size:[{total_batch_val}], Loss: {val_epoch_loss}, Accuracy: top1: {100 * val_epoch_acc}%, top{k}: {100 * val_epoch_acc_top}%')

train_data = (f'Summary: Neural Network:{model_down},\n \
    epochs:{num_epochs},batch:{batch_size}, learning rate: {lr}, Intermediate features: {intfeat}, Dropout:{dropout},\n \
    optimizer: {optimizer} \n \
    Training: Total batch: [{total_batch}], Loss:{epoch_loss}, Accuracy: top1: {100 * epoch_acc}%, top{k}: {100 * epoch_acc_top}% \n \
    Validation: batch size:[{total_batch_val}], Loss: {val_epoch_loss}, Accuracy: {100 * val_epoch_acc}%, top{k}: {100 * val_epoch_acc_top}%')

train_file = open('train_data_file.txt', 'a')
train_file.write(train_data + "\n")

# print(f'Summary loss train: {summary_loss_train}')
# print(f'Summary loss train shape: {np.shape(summary_loss_train)}')
# print(f'Summary loss val: {summary_loss_train}')
# print(f'Summary loss val shape: {np.shape(summary_loss_val)}')
# summary_acc_train_cpu = summary_acc_train.cpu()
#print(f'Summary acc train shape: {np.shape(summary_acc_train.cpu())}')

sommario_acc_train_array = []
for idx in range(len(summary_acc_train)):
    sommario_acc_train_array.append(summary_acc_train[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_train_array[idx])
print(f'Sommario acc train array shape: {np.shape(sommario_acc_train_array)}')

sommario_acc_train_top_array = []
for idx in range(len(summary_acc_train_top)):
    sommario_acc_train_top_array.append(summary_acc_train_top[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_train_array[idx])
print(f'Sommario acc train top array shape: {np.shape(sommario_acc_train_top_array)}')

sommario_acc_val_array = []
for idx in range(len(summary_acc_val)):
    sommario_acc_val_array.append(summary_acc_val[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_val_array[idx])
print(f'Sommario acc val array shape: {np.shape(sommario_acc_val_array)}')

sommario_acc_val_top_array = []
for idx in range(len(summary_acc_val_top)):
    sommario_acc_val_top_array.append(summary_acc_val_top[idx].cpu().clone().detach().numpy())
    # print(sommario_acc_val_array[idx])
print(f'Sommario acc val top array shape: {np.shape(sommario_acc_val_top_array)}')

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
    #print(sommario_acc_train_array[idx])
# print(f'Sommario acc train array shape: {np.shape(sommario_acc_train_array)}')

sommario_acc_train_top_array = []
for idx in range(len(summary_acc_train_top)):
    sommario_acc_train_top_array.append(summary_acc_train_top[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_train_array[idx])
# print(f'Sommario acc train top array shape: {np.shape(sommario_acc_train_top_array)}')

sommario_acc_val_array = []
for idx in range(len(summary_acc_val)):
    sommario_acc_val_array.append(summary_acc_val[idx].cpu().clone().detach().numpy())
    #print(sommario_acc_val_array[idx])
# print(f'Sommario acc val array shape: {np.shape(sommario_acc_val_array)}')

sommario_acc_val_top_array = []
for idx in range(len(summary_acc_val_top)):
    sommario_acc_val_top_array.append(summary_acc_val_top[idx].cpu().clone().detach().numpy())
    # print(sommario_acc_val_array[idx])
# print(f'Sommario acc val top array shape: {np.shape(sommario_acc_val_top_array)}')

ax2.set_title("Accuracy")
ax2.plot(x, sommario_acc_train_array, label='Training Accuracy')
ax2.plot(x, sommario_acc_val_array, label='Validation Accuracy')
ax2.plot(x, sommario_acc_train_top_array, label = 'Training top5 Accuracy')
ax2.plot(x, sommario_acc_val_top_array, label = 'Validation top5 Accuracy')
ax2.legend()

plt.show()