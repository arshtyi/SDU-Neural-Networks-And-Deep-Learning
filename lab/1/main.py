import torch

x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
z = torch.arange(12, dtype=torch.float32).reshape(4, 3)
result = torch.matmul(x, z)
print("x:\n", x)
print("z:\n", z)
print("x @ z:\n", result)
