from typing import Sequence
import numpy as np
import torch
import torch.nn  as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist")

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(f'device = {device}')

#hyperparameters
input_size = 28
Sequence_length = 28
hidden_size = 128
num_classes = 10
num_epoch = 2
batch_size = 64
learning_rate = 0.001

num_layers = 2

#MNIST data
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,  transform=transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset= test_dataset,  batch_size= batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()

#Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) # x -> (batch_size, seq_length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.rnn(x, (h0,c0)) # out: (batch_size, seq_length, hidden_size): (N, 28, 128)
        out = out[:, -1, :]  # (N,128)
        out = self.fc(out)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

#writer.add_graph(RNN(input_size, hidden_size, num_layers, num_classes), (samples.reshape(-1, Sequence_length, input_size), hidden_size), verbose=True)
#writer.close()
#sys.exit()

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

#training loop
n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, Sequence_length, input_size).to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs.data, 1)
        running_correct += (pred == labels).sum().item()
        running_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epoch}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuruacy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0

#test
labels1 = []
preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, Sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

        class_preictions = [nn.functional.softmax(output, dim=0) for output in outputs]
        preds.append(class_preictions)
        labels1.append(labels)

    preds = torch.cat([torch.stack(batch) for batch in preds])
    lables1 = torch.cat(labels1)

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

    classes = range(10)
    for i in classes:
        lables_i = lables1 == i
        preds_i  = preds[:, i]
        writer.add_pr_curve(str(i), lables_i, preds_i, global_step=0)
        writer.close()
    