import torch

x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)

'''torch 사칙연산'''
# z = x + y
z = torch.add(x, y)

print(z)
# z = x - y
z = torch.sub(x, y)

# z = x * y
z = torch.mul(x, y)

# z = x / y
z = torch.div(x, y)


'''inplace'''
y.add_(x) # y 값을 바꿈!
print(y)

'''slicing'''
x = torch.rand(5, 3)
print(x)
print(x[:, 0])
print(x[1, :])
print(x[1, 1]) # tensor
print(x[1, 1].item()) # actual value

'''reshaping'''
x = torch.rand(4, 4)
print(x)
y = x.view(16) # 4 * 4 = 16
print(y)

y = x.view(-1, 8) # 4 * 4 = 16 = 2 * 8
print(y.size())

'''numpy to torch, torch to numpy'''
import numpy as np

# 1. torch to numpy
 
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b))

### 주의 : GPU가 아닌 CPU에서 돌릴 때, 같은 메모리 공간을 공유하고 있음
a.add_(1)
print(a)
print(b) # b도 바뀜!


# 2. numpy to torch
a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

### 주의 : GPU가 아닌 CPU에서 돌릴 때, 같은 메모리 공간을 공유하고 있음
a += 1
print(a)
print(b)


'''cuda toolkit'''
if otrch.cuda.is_available():
    device = torch.device('cuda')
    # cpu to gpu
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device) # move it to the device
    z = x + y # perform on the GPU
    z.numpy() # 에러. numpy can only handle CPU tensors. 따라서 GPU tensor를 numpy로 변환 불가

    # gpu to cpu
    z = z.to("cpu")


'''requires_grad'''
# 최적화를 위해 gradient를 구하고자 할 때 설정
x = torch.ones(5, requires_grad=True)
print(x)