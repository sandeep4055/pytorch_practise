# import libraries
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

# Data Loading

train = datasets.MNIST(root="data/", train=True, transform=transforms.ToTensor(), download=True)
test = datasets.MNIST(root="data/", train=False, transform=transforms.ToTensor(), download=True)

# Dataloader
batch_size = 64

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model


class CNN(nn.Module):

    # Constructor
    def __init__(self, num_channels):
        super(CNN, self).__init__()
        # Convolution block 1
        self.c1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.a1 = nn.ReLU()
        # Max Pool
        self.p1 = nn.MaxPool2d(kernel_size=2)

        # Convolution block 2
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.a2 = nn.ReLU()
        # Max Pool
        self.p2 = nn.MaxPool2d(kernel_size=2)

        # Fully Connected
        self.d1 = nn.Linear(in_features=32 * 7 * 7, out_features=100)
        self.a3 = nn.ReLU()
        self.d2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, data):

        # Convolution-1
        x = self.c1(data)
        x = self.a1(x)
        x = self.p1(x)

        # Convolution 2
        x = self.c2(x)
        x = self.a2(x)
        x = self.p2(x)

        # flattened
        x = x.view(x.shape[0], -1)

        # Fully Connected
        x = self.d1(x)
        x = self.a3(x)
        x = self.d2(x)

        return x


# Initiate Model
model = CNN(num_channels=1)

# Initiate loss & optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model Training
epochs = 10+
iter = 0
for epoch in range(epochs):
    print(f"epoch {epoch} training started")

    for idx, (images, labels) in enumerate(train_loader):
        # forward pass
        outputs = model(images)
        # loss
        loss = criterion(outputs, labels)
        # compute gradients
        optimizer.zero_grad()
        loss.backward()
        # optimizer
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            for test_images, test_labels in test_loader :
                test_outputs = model(test_images)

                # Taking maxing Value index
                argmax_outputs = torch.argmax(test_outputs, dim=1)

                # Compare tensors to get No of correct results
                num_matches = torch.eq(argmax_outputs, test_labels).sum().item()

                correct += num_matches
                total += test_outputs.shape[0]

            accuracy = (correct / total) * 100

            print(f"Accuracy after {iter} iterations is {accuracy}")































