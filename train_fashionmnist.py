# train_fashionmnist.py
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(output, target):
    preds = output.argmax(dim=1)
    return (preds == target).float().mean().item()


# ---------------------------
# Model: CNN-3-128 like baseline
# ---------------------------
class CNN3Baseline(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.5):
        super().__init__()
        # Input: 1x28x28
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(1, 128, kernel_size=3, padding=1),  # 128x28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x14x14

            # conv2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x7x7

            # conv3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),  # 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Train / Validate
# ---------------------------
def train_one_epoch(model, device, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs, labels) * batch_size
        n += batch_size
    return running_loss / n, running_acc / n


def validate(model, device, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy(outputs, labels) * batch_size
            n += batch_size
    return running_loss / n, running_acc / n


# ---------------------------
# Main
# ---------------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
    print("Device:", device)

    # Transforms: include some augmentations for training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(28, scale=(0.9,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets
    data_root = args.data_dir
    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=train_transform)
    val_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = CNN3Baseline(num_classes=10, dropout_p=args.dropout).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)

        scheduler.step(val_acc)

        print(f"  train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
        print(f"  val_loss:   {val_loss:.4f}, val_acc:   {val_acc:.4f}")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.save_dir, f"best_model_epoch{epoch}_acc{val_acc:.4f}.pth")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_acc': val_acc}, ckpt_path)
            print("  Saved best model:", ckpt_path)

        # save last
        last_path = os.path.join(args.save_dir, f"last_model_epoch{epoch}.pth")
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'val_acc': val_acc}, last_path)

    # Plot curves
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "training_curves.png")
    plt.savefig(plot_path)
    print("Saved training plot to", plot_path)
    print("Best val acc:", best_val_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN baseline on Fashion-MNIST")
    parser.add_argument("--data_dir", type=str, default="./data", help="dataset root")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="save directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cpu", action="store_true", help="force use cpu")
    args = parser.parse_args()
    main(args)
