import torch
print(torch.cuda.is_available())
### 看当前pytorch中有哪些工具箱
print(dir(torch))
### 看一下某个工具箱中有哪些工具
print(dir(torch.cuda))
### 看一下工具怎么使用
print(help(torch.cuda.is_available))