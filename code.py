import cv2
import torch
import torch.nn as nn
from torch import optim, cuda
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder, ImageNet
from torchvision import models
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.utils import make_grid
from PIL import Image

#from timeit import default_timer as timer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['font.size'] = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = 'archive/train'
valid_dir = 'archive/test'

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
        T.Compose([
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
        T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Imagenet standards
    ]),
}

#load the train and valid data
train_set = ImageFolder(train_dir,transform = image_transforms['train'])
valid_set = ImageFolder(valid_dir,transform = image_transforms['valid'])

print(f"Length of Train Data : {len(train_set)}")
print(f"Length of Validation Data : {len(valid_set)}")

img, label = train_set[0]
print(img.shape,label)

print('class -> idx : ',train_set.class_to_idx)
# on aura besoin d'un dictionnaire qui fait le sens inverse (idx -> class)
idx_to_class = {train_set.class_to_idx[class_name]: class_name for class_name in  train_set.class_to_idx}
print('idx -> class : ',idx_to_class)

def display_img(img,label):
    print(f"Label : {train_set.classes[label]}")
    plt.imshow(img.permute(1,2,0))

#display the first image in the dataset
display_img(*train_set[0])

batch_size = 128
#load the train and validation into batches.

train_dataloader = DataLoader(train_set, batch_size, shuffle = True, num_workers = 2, pin_memory = True)
valid_dataloader = DataLoader(valid_set, batch_size*2, num_workers = 2, pin_memory = True)

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
        break
        
show_batch(train_dataloader)

class EmotionRecognition(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device) 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss with cross entropy
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.to(device)
        labels = labels.to(device) 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

class Deep_Emotion(EmotionRecognition):
    def __init__(self):
        '''
        Deep_Emotion class contains the network architecture.
        '''
        super(Deep_Emotion,self).__init__()
        self.conv1 = nn.Conv2d(3,10,3).to(device)
        self.conv2 = nn.Conv2d(10,10,3).to(device)
        self.pool2 = nn.MaxPool2d(2,2).to(device)

        self.conv3 = nn.Conv2d(10,10,3).to(device)
        self.conv4 = nn.Conv2d(10,10,3).to(device)
        self.pool4 = nn.MaxPool2d(2,2).to(device)

        self.norm = nn.BatchNorm2d(10).to(device)

        self.fc1 = nn.Linear(810,50).to(device)
        self.fc2 = nn.Linear(50,7).to(device)

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        ).to(device)

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        ).to(device)
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x

    def forward(self,input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = self.conv2(out)
        out = F.relu(self.pool2(out))

        out = F.relu(self.conv3(out))
        out = self.norm(self.conv4(out))
        out = F.relu(self.pool4(out))

        out = F.dropout(out)
        out = out.view(-1, 810)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, valid_dataloader):
    model.to(device)
    model.eval()
    outputs = [model.validation_step(batch) for batch in valid_dataloader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_dataloader, valid_dataloader, opt_func):
    history = []
    parametre_non_geles = []
    for parameter in model.parameters():
        if parameter.requires_grad == True:
            parametre_non_geles.append(parameter)
    optimizer = opt_func(parametre_non_geles, lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_dataloader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, valid_dataloader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

num_epochs = 300
model = Deep_Emotion()
opt_func = optim.SGD
lr = 0.001

#fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_dataloader, valid_dataloader, opt_func)

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
plot_accuracies(history)

def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')

plot_losses(history)

confusion_matrix = np.zeros((7, 7))
with torch.no_grad():
    for i, (images, classes) in enumerate(valid_dataloader):
        images = images.to(device)
        classes = classes.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    print(cm)
    fig = plt.figure(1, figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion_matrix, train_set.classes)

torch.save(model.state_dict(), 'model/model.pth')
