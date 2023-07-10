# VGG16 Pretrained Model- Cifar10 Data - TensorFlow & Keras Framework- Functionnal API

# import needed libraries
import numpy as np

from sklearn.metrics import classification_report

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check if GPU is available & define "device"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load and normalize the CIFAR10 training and test datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainset, valset = torch.utils.data.random_split(trainset, [40000, 10000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=32,
                                        shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

# Define a VGG16 model
model = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 10)])  # Add our layer with 10 outputs
model.classifier = nn.Sequential(*features)  # Replace the model classifier
model = model.to(device)

# Compile the model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4, lr=0.001) 
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, min_lr=0.00005)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    scheduler.step(val_loss)
    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(trainloader)}, Validation Loss: {val_loss / len(valloader)}")

# Test the model
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Classification report
target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_true = []
y_pred = []
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())
print(classification_report(y_true, y_pred, target_names=target_names))
