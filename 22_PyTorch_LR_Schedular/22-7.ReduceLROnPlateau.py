import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10,1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

print(optimizer.state_dict())

for epoch in range(60):
    optimizer.step()
    val_loss = validate(...)
    scheduler.step(val_loss)
    print(optimizer.state_dict()['param_groups'][0]['lr'])

