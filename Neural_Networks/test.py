import torch


tensor1 = torch.rand((64, 10))
tensor2 = torch.rand((64, 10))


print(torch.argmax(tensor1, dim=1))
print(torch.argmax(tensor2, dim=1))
print(torch.eq(torch.argmax(tensor1, dim=1), torch.argmax(tensor2, dim=1)).sum().item())