import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 4
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download = True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                            download = True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                        shuffle=True) 

classes = ('plane', 'car', 'bird', 'cat', 'deer','dog','frog','horse','ship','truck')

class ConvNet(nn.Module):
    def __init__(self, in_channels=3, out_channel_list=[6, 16], kernel_size=5):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channel_list[0], kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=out_channel_list[0], out_channels=out_channel_list[1], kernel_size=kernel_size)
        self.fc1 = nn.Linear(out_channel_list[1] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        outs = self.maxpool(self.relu(self.conv1(x)))
        outs = self.maxpool(self.relu(self.conv2(outs)))
        # flatten
        outs = outs.view(-1, 16 * 5 * 5)
        outs = self.relu(self.fc1(outs))
        outs = self.relu(self.fc2(outs))
        outs = self.fc3(outs)
        return outs
        

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape : [4, 3, 32, 32] = 4, 3, 1024
        # input_layer : 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1} /{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finish Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max returns (values, index)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (labels==predictions).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label==pred):
                n_class_correct[label] +=1
            n_class_samples[label]+=1

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy of the network: {acc} % ")
