# https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html#configure-the-validation-logic
import os

import tensorboard
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer

# hyper parameters
input_size = 784 # 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001
split_ratio = 0.8

# 1) Model
class LITNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LITNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(),lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        # reshape
        images = images.reshape(-1, 28 * 28).to(self.device)
        labels = labels.to(self.device)

        # forward
        outputs = self(images)

        # loss
        loss = F.cross_entropy(outputs, labels)
        tensorboard_logs = {'train_loss': loss}

        return {'loss' : loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=4,
                        shuffle=True)
        return train_loader


    # model.eval() and torch.no_grad() are called automatically for validation.
    # trainer.validate() loads the best checkpoint automatically by default if checkpointing was enabled during fitting.
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        images = images.reshape(-1, 28 * 28).to(self.device)
        labels = labels.to(self.device)

        y_hat = self(images)
        val_loss = F.cross_entropy(y_hat, labels)
        self.log("val_loss", val_loss)
        return {'val_loss': val_loss}

    def val_dataloader(self):
        valid_dataset = torchvision.datasets.MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
        valid_loader = DataLoader(valid_dataset,
                        batch_size=batch_size,
                        num_workers=4,
                        shuffle=False) # 실수로 True로 하면 warning 뜸
        return valid_loader

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # # model.eval() and torch.no_grad() are called automatically for testing.
    # # trainer.test() loads the best checkpoint automatically by default if checkpointing is enabled.
    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = F.cross_entropy(y_hat, y)
    #     self.log("test_loss", test_loss)

if __name__=="__main__":
    # fast_dev_run=True : this will run a single batch training and also throgh validation if you have a validation step. You can test if this model works.
    # auto_lr_find= True : this will run the algorithm to find the best learning rate
    trainer = Trainer(auto_lr_find=True, max_epochs=num_epochs)#,fast_dev_run=True)
    
    model = LITNeuralNet(input_size, hidden_size, num_classes)
    trainer.fit(model)
