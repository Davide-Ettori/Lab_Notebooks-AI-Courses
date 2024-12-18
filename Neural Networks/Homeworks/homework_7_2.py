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
import matplotlib.pyplot as plt

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available() and False
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

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# *****************************************************************************
# Building the neural network
# *****************************************************************************
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 20, 4, 1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.Conv2d(20, 20, 4, 2),
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
        self.Decoder = nn.Sequential(
            nn.Linear(10, 360),
            nn.ReLU(),
            nn.BatchNorm1d(360),
            nn.Dropout(0.1),
            nn.Linear(360, 500),  # Added layer
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Dropout(0.1),
            nn.Linear(500, 720),
            nn.ReLU(),
            nn.BatchNorm1d(720),
            nn.Dropout(0.1),
            nn.Unflatten(1, (20, 6, 6)),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.Upsample(scale_factor=2, mode='bicubic'),
            nn.ConvTranspose2d(20, 20, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(20, 1, kernel_size=4, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )
        self.flatten = nn.Flatten()

    def forward(self, x, enc_mode=1):
        #print(f"x: {x.shape}")
        z = self.Encoder(x)
        #print(f"z: {z.shape}")
        z2 = enc_mode * z + (2 - enc_mode) * torch.randn(z.shape).to(device)
        #print(f"z2: {z2.shape}")
        f = self.Decoder(z2)
        #print(f"f: {f.shape}")
        e = f - x
        e = self.flatten(e)
        e = torch.cat([z, e], dim=1)
        return e

model = NeuralNetwork().to(device)

# *****************************************************************************
# Train and test loops
# *****************************************************************************
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, _) in enumerate(dataloader):
        pred = model(X.to(device))
        loss = loss_fn(pred, X.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for X, _ in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, X.to(device)).item()
    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n {size} {num_batches}")

# *****************************************************************************
# Loss function
# *****************************************************************************
class GaussianAutoencoderLoss(nn.Module):
    def forward(self, pred, target):
        z = pred[:, :10]
        e = pred[:, 10:]
        loss = 0.5 * torch.norm(z, dim=1).mean() + 0.5 * torch.norm(e, dim=1).mean()
        return loss

loss_fn = GaussianAutoencoderLoss()

learning_rate = 0.25
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# *****************************************************************************
# Standard training epochs
# *****************************************************************************
print(model)
print("Training model...")
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

# *****************************************************************************
# Generating new images using the learned autoencoder
# *****************************************************************************
for s in range(20):
    x = model(torch.zeros((1, 1, 28, 28)).to(device), enc_mode=0)
    imgX = x[0, 10:].reshape(28, 28).detach().to("cpu")
    plt.imshow(imgX, cmap='gray')
    plt.show()
