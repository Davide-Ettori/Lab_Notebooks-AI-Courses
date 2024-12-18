# *****************************************************************************
# *****************************************************************************
# Gaussian Autoencoder
# *****************************************************************************
# *****************************************************************************

# *****************************************************************************
# Preamble and dataset loading, based on PyTorch tutorial
# *****************************************************************************
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import numpy as np
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(device)
print(f"Using {device} device")

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = ???? #!!! Fill in !!!#

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# *****************************************************************************
# Building the neural network
# *****************************************************************************
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential( # Same as HW6's CNN
            nn.Conv2d(1,20,4,1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.Conv2d(20,20,4,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.BatchNorm1d(250),
            nn.Dropout(0.1),
            nn.Linear(250, 10)
        )
        self.Decoder = nn.Sequential( #!!! Fill in !!!# "Invert" the encoder
            nn.Linear(????),
            nn.ReLU(),
            nn.BatchNorm1d(????),
            nn.Dropout(0.1),
            nn.Linear(????),
            nn.ReLU(),
            nn.Unflatten(1, ????),
            nn.BatchNorm2d(????),
            nn.Dropout(0.1),
            nn.Upsample(scale_factor=????, mode='bicubic'),
            nn.ConvTranspose2d(????),
            nn.ReLU(),
            nn.BatchNorm2d(????),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(????),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x, enc_mode = 1):
        z = ???? #!!! Fill in !!! Encoder, giving embedding
        z2 = enc_mode*z + (2-enc_mode)*torch.randn(z.shape) # Adding noise
        f = ???? #!!! Fill in !!! Decoder, giving reconstructed x
        e = ???? #!!! Fill in !!! Reconstruction error
        e = self.flatten(e)
        e = torch.cat(????,dim=1) #!!! Fill in !!! Concatenate embedding and reconstruction error
        return e

model = NeuralNetwork().to(device)

# *****************************************************************************
# Train and test loops
# *****************************************************************************
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = ???? #!!! Fill in !!!

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += ???? #!!! Fill in !!!
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n {size} {num_batches}")

# *****************************************************************************
# Optimization prameters and initialization
# *****************************************************************************
loss_fn = ???? #!!! Fill in !!!
learning_rate = ???? #!!! Fill in !!!#
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# *****************************************************************************
# Standard training epochs
# *****************************************************************************
print(model)
print("Training model...")
epochs = ???? #!!! Fill in !!!#
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# *****************************************************************************
# Generating new images using the learned autoencoder
# *****************************************************************************
for s in range(20):
    x = ???? #!!! Fill in !!! Generate a new image by calling the model with a 0 input and argument enc_mode = 0
    imgX = ????.reshape(????).detach().to("cpu") #!!! Fill in !!! Extract the image part of x
    plt.imshow(imgX)
    plt.show()
