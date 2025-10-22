import os
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# --------------------------- Utilities ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------- SE Module ---------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), -1, 1, 1)
        return x * w

# --------------------------- ResNetSE for 28x28 input ---------------------------
class ResNetSE(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(weights=None)  # 不加载预训练
        # 修改输入通道
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # 去掉最大池化，28x28 输入太小
        # 在每个 block 后加入 SE
        for name, module in self.backbone.named_children():
            if isinstance(module, nn.Sequential):
                for i, block in enumerate(module):
                    block.add_module("se", SEBlock(block.conv2.out_channels))
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# --------------------------- MixUp ---------------------------
def mixup_data(x, y, alpha=0.4, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --------------------------- Training / Validation ---------------------------
def train_one_epoch(model, device, dataloader, criterion, optimizer, scheduler=None, mixup_alpha=0.4):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        if mixup_alpha > 0:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha, device=device)
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            preds = outputs.argmax(dim=1)
            acc = (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).mean().item()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        batch_size = imgs.size(0)
        running_loss += loss.item() * batch_size
        running_acc += acc * batch_size
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
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            n += batch_size
    return running_loss / n, running_acc / n

# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transforms
    train_transform = transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.FASHION_MNIST),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Datasets / Loaders
    train_dataset = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=train_transform)
    val_dataset = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = ResNetSE(num_classes=10).to(device)
    print("Model params:", sum(p.numel() for p in model.parameters()))

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr * 4,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Training Loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer, scheduler, mixup_alpha=0.4)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnetse_best.pth")
    print("Best val acc:", best_acc)

if __name__ == '__main__':
    main()
