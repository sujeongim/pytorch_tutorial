# https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html?highlight=lambdalr#torch.optim.lr_scheduler.LambdaLR
# https://sanghyu.tistory.com/113

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10,1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

lambda1 = lambda epoch : epoch / 10
scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)

print(optimizer.state_dict())

for epoch in range(5):
    optimizer.step()
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])

