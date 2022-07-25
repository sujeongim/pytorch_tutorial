from tkinter import NE
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform = transform,
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform = transform,
                                           download=True)     


# dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size)

# sample test
examples = iter(test_loader)
example_data, example_targets = examples.next()
print(example_data.shape, example_targets.shape)
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0], cmap='gray')
#plt.show()
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image("mnist_images",img_grid)
writer.close()
#sys.exit()

# model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        outs = self.l1(x)
        outs = self.relu(outs)
        outs = self.l2(outs)
        return outs
model = NeuralNet(input_size, hidden_size, num_classes)


# loss, optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters, lr=learning_rate)

writer.add_graph(model, example_data.reshape(-1, 28 * 28))
writer.close()
#sys.exit()

# training loop
# 3) Training Loop
train_loss = []
valid_loss = []
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    model.train()
    total_loss = 0
    print("Train")
    for i, (images, labels) in enumerate(trainloader):
        # reshape
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)

        # loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        
        # reset
        optimizer.zero_grad()
        # backprop
        loss.backward()
        # update
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predictions = torch.max(outputs, 1)
        running_correct += (predictions==labels).sum().item()

        # print
        if (i+1) % 100 ==0:
            print(f"iter {epoch+1}/{num_epochs}, step {i+1} /{n_total_steps} : train_loss = {loss.item():.4f}")
            writer.add_scalar('training_loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('training_acc', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
    print(f"avg_train_loss={total_loss/len(train_loader):.4f}")
    train_loss.append(total_loss/len(train_loader))
    
    
    print("Eval")
    pred_labels = []
    preds = []
    with torch.no_grad():
        model.eval()
        total_loss_val = 0
        n_samples = 0
        n_correct = 0
        for j, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)

            # predictions # return : max_value, max_index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions==labels).sum().item()

            class_predictions = [F.softmax(output, dim=0) for output in outputs]
            preds.append(class_predictions)
            pred_labels.append(predictions)

            # loss
            loss_val = criterion(outputs, pred_labels)
            total_loss_val+=loss_val.item()
        
        preds = torch.cat([torch.stack(batch) for batch in preds])
        pred_labels = torch.cat(pred_labels)

        acc = 100.0 * n_correct / n_samples
        print(f"avg_valid_loss={total_loss_val/len(test_loader):.4f}, acc = {acc}")
        valid_loss.append(total_loss_val/len(test_loader))

        classes  = range(10)
        for i in classes:
            labels_i = labels == i
            print(labels_i)
            preds_i = preds[:, i]
            writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)

            # 3) Training Loop
train_loss = []
valid_loss = []
n_total_steps = len(trainloader)
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    model.train()
    total_loss = 0
    print("Train")
    for i, (images, labels) in enumerate(trainloader):
        # reshape
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)

        # loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        
        # reset
        optimizer.zero_grad()
        # backprop
        loss.backward()
        # update
        optimizer.step()
        
        
        # print
        if (i+1) % 100 ==0:
            print(f"iter {epoch+1}/{num_epochs}, step {i+1} /{n_total_steps} : train_loss = {loss.item():.4f}")
    print(f"avg_train_loss={total_loss/len(trainloader):.4f}")
    train_loss.append(total_loss/len(trainloader))
    
    
    print("Eval")
    with torch.no_grad():
        model.eval()
        total_loss_val = 0
        n_samples = 0
        n_correct = 0
        for j, (images, labels) in enumerate(validloader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)

            # predictions # return : max_value, max_index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions==labels).sum().item()

            # loss
            loss_val = criterion(outputs, labels)
            total_loss_val+=loss_val.item()
        
        acc = 100.0 * n_correct / n_samples
        print(f"avg_valid_loss={total_loss_val/len(validloader):.4f}, acc = {acc}")
        valid_loss.append(total_loss_val/len(validloader))
 
torch.save(model.state_dict(), "mnist_ffn.pth")