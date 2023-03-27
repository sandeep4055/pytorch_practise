import torch

# ================================================= #
#               Tensor indexing                     #
# ================================================= #

tensor = torch.rand(size=(10, 12))

print(tensor[:, :])     # [rows, columns]
print(tensor[:, 0])     # returns first element in all rows
print(tensor[:2, :5])   # returns first five elements in first two rows

# Advance indexing
tensor_2 = torch.arange(10)
print(tensor_2)

print(tensor_2[tensor_2 < 2])
print(tensor_2[(tensor_2 < 3) | (tensor_2 > 7)])    # less than 2 or greater than 7
print(tensor_2[(tensor_2 < 3) & (tensor_2 > 7)])    # less than 2 and greater than 7
print(tensor_2[tensor_2.remainder(2) == 0])         # x/2==0
print(torch.where(tensor_2 > 5, tensor_2, tensor_2**2))     # where greater than 5 stays else squared
print(torch.tensor([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7]).unique())
print(tensor_2.numel())     # no of elements






