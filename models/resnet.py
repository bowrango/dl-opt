import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Data preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet expects 3 channels
    transforms.Resize(224),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load Fashion MNIST dataset
trainData = datasets.FashionMNIST(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

testData = datasets.FashionMNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)

batchSize = 256
trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True)
testLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)

criterion = nn.CrossEntropyLoss()

class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Block(nn.Module):
    # Basic Block for ResNet-18/34
    expansion = 1
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        return out
    
class ResNet18(nn.Module):
    # https://arxiv.org/pdf/1512.03385

    # 1. Stem (1 conv = 1 layer)
    # 2. Residuals (4 stages × 2 blocks × 2 convs = 16 layers)
    # 3. Classifier  (1 fc = 1 layer)

    def __init__(self, num_classes=10, d=64):
        super(ResNet18, self).__init__()

        # Stem
        self.conv1 = nn.Conv2d(3, d, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residuals
        self.layer1 = self._make_layer(d, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Block.expansion, num_classes)

    def _make_layer(self, in_ch, out_ch, blocks, stride):
    #       4 stages (layer1 … layer4)
    #       Each stage has 2 Blocks
    #       Each Block has 2 convolutions
    #       4 stages × 2 blocks × 2 convs = 16 layers
        downsample = None
        if stride != 1 or in_ch != out_ch*Block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch*Block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch*Block.expansion),
            )

        layers = [Block(in_ch, out_ch, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(Block(out_ch, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residuals
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet18(num_classes=10).to(device)

    learning_rate = 0.0001
    weight_decay = 0.01  # L2 regularization lambda
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_accuracies = []
    test_accuracies = []

    early_stopper = EarlyStopper(patience=5)
    numEpochs = 10

    for epoch in range(numEpochs):
        print(f"Epoch {epoch+1}/{numEpochs}")

        # Training
        model.train()
        train_correct = 0
        train_total = 0
        for data, label in trainLoader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            print(loss)

            _, predicted = torch.max(logits.data, 1)
            train_total += label.size(0)
            train_correct += (predicted == label).sum().item()

            loss.backward()
            optimizer.step()

        train_accuracy = 100 * train_correct / train_total

        # Testing
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        with torch.no_grad():
            for data, label in testLoader:
                data, label = data.to(device), label.to(device)
                logits = model(data)
                loss = criterion(logits, label)
                test_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                test_total += label.size(0)
                test_correct += (predicted == label).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_loss = test_loss / len(testLoader)

        # Store results
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(f"  Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Plot results after epoch
    plt.figure(figsize=(12, 6))
    plt.plot(range(numEpochs), train_accuracies, 'b-', linewidth=2, label='Train Accuracy', marker='o')
    plt.plot(range(numEpochs), test_accuracies, 'r-', linewidth=2, label='Test Accuracy', marker='s')

    # Add epoch markers
    for epoch in range(numEpochs):
        plt.axvline(x=epoch, color='black', linestyle='--', alpha=0.3)

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training and Test Accuracy vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    plt.xlim(-0.5, numEpochs - 0.5)
    plt.show()