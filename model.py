import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

transform = transforms.Compose(
    [transforms.Grayscale(),
     transforms.Resize((24, 24)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.ImageFolder(root='C:/Users/Public/dataset_new/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

valid_dataset = datasets.ImageFolder(root='C:/Users/Public/dataset_new/test', transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*18*18, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.dropout1(x)
        x = x.view(-1, 64*18*18)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.softmax(x, dim=0)
        return output

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss = []
valid_accuracy = []
for epoch in range(15):
    running_loss = 0.0
    num_batches=1
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(inputs.shape)
        # print(outputs.shape)
        # print(labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches+=1
        if i % 50 == 49:
            # train_loss.append(running_loss / 50)
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    epoch_loss = running_loss / num_batches
    train_loss.append(epoch_loss)
    print('[Epoch %d] training loss: %.3f' % (epoch + 1, epoch_loss))
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    valid_accuracy.append(accuracy)
    print('Accuracy of the network on the validation set: %d %%' % (100 * correct / total))

import matplotlib.pyplot as plt

# Plot and save the loss curve
plt.plot(train_loss)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('train_loss.png')
plt.show()

# Plot and save the accuracy curve
plt.plot(valid_accuracy)
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig('valid_accuracy.png')
plt.show()

torch.save(model.state_dict(), 'models/cnnCat21.pt')
