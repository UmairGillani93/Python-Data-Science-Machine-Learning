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

BATCH_SIZE = 64

# create a data loaders
train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

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
        self.Flatten = nn.Flatten()
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

print(model)

# defining loss function and optimzer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

print(optimizer)
