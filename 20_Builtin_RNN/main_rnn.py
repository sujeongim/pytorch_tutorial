import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

input_size = 28
sequence_length = 28
num_layers = 2


train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # x => (batch_size, seq, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)



    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        # |out| = (batch_size, seq_len, hidden_size)
        # out (N, 28, 128)
        out = out[:, -1, :]
        # out (N, 128)
        y = self.fc(out)
        return y

model = RNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
valid_loss = []
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28, 28).to(device)
        labels = labels.to(device)
        
        # forward
        prediction = model(images)

        # loss 
        loss = criterion(prediction, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"{epoch}/{num_epochs} : loss = {total_loss/len(train_loader)}")

    with torch.no_grad():
        model.eval()
        total_loss_val = 0
        n_samples = 0
        n_correct = 0
        for j, (images, labels) in enumerate(test_loader):
            images = images.view(-1, 28, 28).to(device)
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
        print(f"avg_valid_loss={total_loss_val/len(test_loader):.4f}, acc = {acc}")
        valid_loss.append(total_loss_val/len(test_loader))