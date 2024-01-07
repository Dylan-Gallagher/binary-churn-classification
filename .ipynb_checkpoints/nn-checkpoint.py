import pandas as pd

# Load the data
data = pd.read_csv('data/train.csv').drop(['id', 'CustomerId', 'Surname'], axis=1)
X = pd.get_dummies(data.drop(['Exited'], axis=2), columns=['Geography', 'Gender'], dtype=float)
y = data['Exited']

from sklearn.preprocessing import StandardScaler
# Normalize the data
columns_to_normalize = ["CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure"]
scaler = StandardScaler()
X[columns_to_normalize] = scaler.fit_transform(X[columns_to_normalize])

from sklearn.model_selection import train_test_split

# Make train-dev split
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2)
test_data = pd.read_csv('data/test.csv')
X_test = pd.get_dummies(test_data.drop(['CustomerId', 'Surname'], axis=1), columns=['Geography', 'Gender'], dtype=float)

import torch
from torch.utils.data import Dataset, DataLoader


# Create the dataset
class ChurnDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values)
        self.labels = torch.tensor(labels.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = ChurnDataset(X_train, y_train)
dev_dataset = ChurnDataset(X_dev, y_dev)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

# Make a neural network
class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return F.sigmoid(x)
    

model = NNet()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    test_loop(dev_loader, model, loss_fn)
print("Done!")