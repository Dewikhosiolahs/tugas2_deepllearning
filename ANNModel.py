# Import library yang diperlukan
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Definisi model ANN
class ANNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)  # Fully connected layer 1
        self.fc2 = nn.Linear(512, 256)  # Fully connected hidden layer 2
        self.fc3 = nn.Linear(256, num_classes)  # Output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Trainer untuk model
class ANNTrainer:
    def __init__(self, model, train_loader, test_loader, criterion=nn.CrossEntropyLoss(), lr=0.01,
                 optimizer_type="SGD", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(self.device)
        if optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer_type == "MiniBatchGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_type == "GD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0)
        else:
            raise ValueError("Optimizer type not recognized")

    def train(self, epochs=10):
        self.model.train()
        training_errors = []
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)  # Pindahkan ke GPU/CPU
                images = images.view(images.size(0), -1)  # Flatten gambar sebelum masuk model
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            training_errors.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        return training_errors

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = 100 * correct / total
        print(f"Testing Accuracy: {accuracy:.2f}%")
        return accuracy, all_preds, all_labels