import torch
import numpy as np

# ========================================== #
#           Initialize Tensors               #
# ========================================== #

device = "cuda" if torch.cuda.is_available() else "cpu"

tensor = torch.tensor(data=[[1, 2, 3], [3, 4, 5]], dtype=torch.float32, device=device, requires_grad=True)

print(tensor)
print(tensor.dtype)
print(tensor.device)
print(tensor.requires_grad)

# Other Initialization

x = torch.empty(size=(3, 4))
print(x)
print(x.device)

zeros = torch.zeros(size=(3, 3))
print(zeros)

ones = torch.ones(size=[5, 5])
print(ones)
print(ones.shape)
print(ones.ndim)

rand = torch.rand(size=(4, 4), dtype=torch.float32)  # torch.rand only gives float values not the int values
print(rand)

ranger = torch.arange(start=10, end=1000, step=101)
print(ranger)
print(ranger.shape)
print(ranger.ndim)

linspace_tensor = torch.linspace(start=10, end=1000, steps=10)
print(linspace_tensor)

# uniform distribution

uniform = torch.empty(size=(5, 5)).uniform_(0, 1)
print(uniform)

# Normal distribution

normal = torch.empty(size=(4, 4)).normal_(mean=10, std=1)
print(normal)

# identity matrix

identity = torch.eye(5, 5)
print(identity)

# diagonal matrix >> i.e all elements in diagonal are numbers else zeros

diagonal = torch.diag(normal)
print(diagonal)

# convert tensor types

tensor = torch.arange(start=0, end=5)
print(tensor)
print(tensor.bool())
print(tensor.short())  # int16
print(tensor.long())   # int64
print(tensor.half())   # float16
print(tensor.float())  # float64

# Numpy to tensor
np_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.from_numpy(np_array)
print(np_array)
print(tensor)
print(tensor.numpy())   # similar to tf




