# ImageFolder
# Schedular
# Transfer Learning

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    "train" : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)]),
}

# import data
data_dir = 'data/hymenoptera_data'
sets = ['train', 'val']
image_datasets = {x : datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in sets}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in sets}

dataset_sizes = {x: len(image_datasets[x] for x in sets)}
class_names = image_datasets['train'].classes
print(class_names)

def train_model(model, criterion, optimizer, schedular, num_epoch=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs+1}")
        print("-"* 10)

        for phase in ['train', 'val']:
            if phase =='train':
                model.train()
            elif phase =='val':
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                with torch.set_grad_enables(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase =='train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0) # 왜 곱해?
                running_corrects += torch.sum(preds == labels.data)

            if phase=='train':
                schedular.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss : {epoch_loss:.4f}, Acc : {epoch_acc:.4f}')

            if phase=='val' and epoch_acc > best_acc:
                bset_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60 , time_elapsed % 60))
    print("Best val Acc : {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model

                

######

# get pretrained model
model = models.renet18(pretrained=True)

# parameter freezing
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Scheduler
# : update learning rate
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#for epoch in range(100):
#    train()
#    evaluate()
#    scheduler.step()

model = train_model(model, criterion, optimizer, schedular, num_epochs=20)
