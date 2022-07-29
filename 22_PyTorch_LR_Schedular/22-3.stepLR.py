# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10,1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print(optimizer.state_dict())

for epoch in range(60):
    optimizer.step()
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])