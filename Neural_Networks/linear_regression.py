# import libraries
import torch
import numpy as np

# Input and Outputs

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

output = np.array([56, 81, 119, 22, 103], dtype='float32')

input_tensor = torch.from_numpy(inputs)
output_tensor = torch.from_numpy(output)

# linear function y = wx + b
w = torch.rand(size=(1, 3), requires_grad=True)
b = torch.rand(1, requires_grad=True)

def linear(x):
    yi = w @ x.t() + b
    return yi


# y = linear(input_tensor)

# Loss function mse
def mse_loss(real, pred):
    sub = real-pred
    sqr = sub*sub
    add = torch.sum(sqr)

    return add/real.numel()


"""
loss = mse_loss(output_tensor, y)

# Gradients computing with respective loss
loss.backward()

# Update Weights

with torch.no_grad():   # while updating weights there is no use of gradient tracking or computing/ changing
    w -= w.grad * 1e-3
    b -= b.grad * 1e-3  # 1e-5 is learning rate
"""

# Train for 100 epochs
for i in range(200):
    preds = linear(input_tensor)
    loss = mse_loss(preds, output_tensor)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

# Calculate loss
preds = linear(input_tensor)
loss = mse_loss(preds, output_tensor)
print(loss)
