import torch

# ================================================= #
#               Tensor Reshaping                    #
# ================================================= #

x = torch.arange(10)
x_5x2 = x.view(5, 2)
print(x_5x2)

print(x.reshape(5, 2))

print(x_5x2.stride())
print(x_5x2.is_contiguous())
print(x.reshape(5, 2).is_contiguous())

# view and reshape store the data in contiguous way but if we change the data in view the data in originaltensor changes
# but this doesn't happen in reshape

# Transpose
x_t = x_5x2.t()
print(x_5x2)
print(x_t)
print(x_t.is_contiguous())      # transpose doesnt store data in contiguous way

# concatenation
t1 = torch.rand((2, 5))
t2 = torch.rand((2, 5))

print(torch.cat((t1, t2), dim=0))
print(torch.cat((t1, t2), dim=1))

# flatten
print(t1.view(-1))
print(t1.reshape(-1))

# flatten in batch
batch = 16
t3 = torch.rand(batch, 10, 20)
print(t3.reshape(batch, -1))
print(t3.reshape(batch, -1).shape)

# transpose for higher dimensions we use permute
t4 = t3.permute(0, 2, 1)  # keeping 0 dimension as same since it is batch size we change rows and columns dimensions
print(t3.shape)
print(t4.shape)

# we also have squeeze and un squeeze  for reducing and expanding dimensions

