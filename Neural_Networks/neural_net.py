# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch.optim as optim


# model creation

class NN(nn.Module):
    def __init__(self, input_shape, output):
        super(NN, self).__init__()
        self.l1 = nn.Linear(in_features=input_shape, out_features=50)
        self.l2 = nn.Linear(in_features=50, out_features=output)

    def forward(self, data):
        x = F.relu(self.l1(data))
        x = self.l2(x)

        return x


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameters
input_size = 28*28
batch_size = 32
num_classes = 10
learning_rate = 0.001
epochs = 10

# Load data

train = datasets.MNIST(root="data/", train=True, transform=transform.ToTensor(), download=True)
train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test = datasets.MNIST(root="data/", train=False, transform=transform.ToTensor(), download=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

# Initialize model

model = NN(input_shape=input_size, output=num_classes).to(device)

# loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network

for epoch in range(epochs):

    for batch_idx, (data, targets) in enumerate(train_loader):

        # to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # reshape
        data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()

    print(f"Loss from epoch {epoch}, is :{loss} ")


def check_accuracy(loader, train_model):
    # Set model to evaluation mode
    train_model.eval()

    # Calculate accuracy on test set
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:

            # to cuda
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            # reshape
            inputs = inputs.reshape(inputs.shape[0], -1)

            # Forward pass
            outputs = model(inputs)
            # Get predicted class labels
            _, predicted = torch.max(outputs.data, 1)
            # Count number of correct predictions
            correct += (predicted == labels).sum().item()
            # Count total number of examples
            total += labels.size(0)

        # Calculate overall accuracy
        accuracy = correct / total
        print('Test accuracy: {:.2f}%'.format(100 * accuracy))


check_accuracy(loader=test_loader, train_model=model)














