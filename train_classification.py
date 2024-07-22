import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from homework.models import Classifier, save_model, load_model,  INPUT_MEAN, INPUT_STD
from homework.datasets.classification_dataset import load_data
from pathlib import Path
from torchvision import transforms
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.Inf

    def __call__(self, val_loss, model, model_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, model_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, model_name):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model)
        self.best_loss = val_loss

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Training Loss: {running_loss / len(train_loader)}")

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_loader)
    accuracy = correct / len(val_loader.dataset)
    print(f"Validation Loss: {val_loss}, Accuracy: {accuracy * 100}%")
    return val_loss, accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=INPUT_MEAN, std=INPUT_STD),
    ])
    
    train_loader = load_data('classification_data/train', batch_size=32, shuffle=True)
    val_loader = load_data('classification_data/val', batch_size=64, shuffle=False)
    
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    early_stopping = EarlyStopping(patience=20, verbose=True)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train(model, device, train_loader, optimizer, criterion)
        val_loss, accuracy = validate(model, device, val_loader, criterion)
        
        scheduler.step()
        
        early_stopping(val_loss, model, 'improved_classifier')
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    main()