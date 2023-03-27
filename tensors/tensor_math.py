import torch

# ================================================= #
#           Tensors Math & Comparison operations    #
# ================================================= #

x = torch.tensor(data=[10, 20, 30])
y = torch.tensor(data=[90, 80, 70])

# Addition
z1 = x + y
z2 = torch.add(x, y)
print(z1)
print(z2)

# divide
z3 = torch.divide(x, y)
print(z3)

z4 = torch.true_divide(x, y)
print(z4)

# Inplace operations i.e t = t + x >> t+=x in torch t.add_(x)
# Anything ends with "_" is inplace operation

t = torch.ones(3)
print(t)
t.add_(x)
print(t)

# Exponentiation
exp_x = x**2
print(exp_x)
exp_x = torch.pow(x, 2)
print(exp_x)
print(x.pow(2))

# Matrix multiplication
m = torch.rand(size=(3, 3))
n = torch.rand(size=(3, 3))
z5 = torch.mm(m, n)
print(z5)
z6 = m.mm(n)
print(z6)

# Matrix Exponentiation
print(m)
print(m.matrix_power(3))

# Element wise multiply
print(m * n)
print(z6)

# dot product >>> Element wise multiply and sum them
z7 = torch.dot(x, y)
print(z7)

# Batch Matrix Multiplication >>> if the dimensions is greater than 2 then we use batch matrix multiplication

batch = 32
n = 10
m = 20
p = 30

tensor_1 = torch.rand(batch, n, m)
tensor_2 = torch.rand(batch, m, p)

batch_tensor = torch.bmm(tensor_1, tensor_2)
print(batch_tensor.shape)   # (batch, n, p)

# torch supports to broadcasting similar to numpy and tensorflow

# Other tensor operations

k = torch.tensor(data=[[10, 20, 30],
                       [10, 20, 30],
                       [10, 20, 30]])

k1 = torch.sum(k)
print(k1)
k2 = torch.sum(k, dim=0)
print(k2)
k3 = torch.sum(k, dim=1)
print(k3)

# other operations are max, min, argmax(gives indices), mean(only takes float as input but not int), std(only float),
# eq(elements are equal between two tensors returns booleans)
# sort
# clamp >> clamp the values to given element
print(torch.clamp(k, min=20))   # replace all elements less than 20 to 20 similarly for max



