import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

training_dataset = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_dataset = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)


print("length of training data: %s" %(len(training_dataset)))
print("length of test data: %s" %(len(test_dataset)))

batch_size = 64

# create a data loaders
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

for X,y in test_dataloader:
    print('Tensor X shape : [N, C, H, W]: {}' .format(X.shape))
    print('Tensor y shape : [N, C, H, W]: {}' .format(y.shape, y.dtype))
    break


device = "cuda" if torch.cuda.is_available() else "cpu"
print("using {} device ".format(device))

# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        # we define layers in __init__() method and data flow in "forward" method
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                                nn.Linear(28*28, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512, 10),
                                nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)

# print(model)

# defining loss function and optimzer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# print(optimizer)


# train the model
def train(dataloader, model, loss_fn, optiimzer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # print(f"batch: {batch}")
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# check the model's preformace agains test dataset
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# train the model
EPOCHS = 20

for t in range(EPOCHS):
    print(f"Epoch: {t+1}\n-----------------")
    train(train_dataloader,model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")
