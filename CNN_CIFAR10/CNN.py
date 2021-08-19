import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
num_epoch = 2
batch_size = 4
learning_rate = 0.001

#dataset 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,  transform=transform, download=True)
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()
#print(samples.shape, labels.shape)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Model
PATH ='./cifar.pth'
model = ConvNet().to(device)
model.load_state_dict(torch.load(PATH)) # load Model

#loss and optimization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#traning loop
n_total_steps = len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], step [{i+1}/{n_total_steps}], loss {loss.item():.4f}')
#save
torch.save(model.state_dict(), PATH)


#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    count = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, prediction = torch.max(outputs, 1) 
        n_samples += labels.size(0)
        n_correct += (prediction == labels).sum().item()

        for i in range(batch_size):
            count += 1
            #print(count, n_total_steps ,labels[i])
            n_class_samples[labels[i]] += 1
            if(labels[i] == prediction[i]):
                n_class_correct[labels[i]] += 1
        
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network = {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
            

