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
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
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

# --------------------------- MixUp / CutMix ---------------------------
def mixup_data(x, y, alpha=0.4, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0, device='cpu'):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size, C, H, W = x.size()
    index = torch.randperm(batch_size).to(device)

    # 随机裁剪区域
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x_new = x.clone()
    x_new[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[index]
    return x_new, y_a, y_b, lam_adjusted

def criterion_mix(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --------------------------- Training / Validation ---------------------------
def train_one_epoch(model, device, dataloader, criterion, optimizer, scheduler=None,
                    mixup_alpha=0.4, cutmix_alpha=1.0):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for imgs, labels in tqdm(dataloader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)

        # 随机选择 MixUp 或 CutMix
        r = np.random.rand()
        if cutmix_alpha > 0 and r < 0.5:
            imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, alpha=cutmix_alpha, device=device)
            outputs = model(imgs)
            loss = criterion_mix(criterion, outputs, labels_a, labels_b, lam)
            preds = outputs.argmax(dim=1)
            acc = (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).mean().item()
        elif mixup_alpha > 0:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha, device=device)
            outputs = model(imgs)
            loss = criterion_mix(criterion, outputs, labels_a, labels_b, lam)
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

# --------------------------- Ensemble Inference ---------------------------
def ensemble_predict(models, device, dataloader):
    for model in models:
        model.eval()
    preds_all = []
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Ensemble"):
            imgs = imgs.to(device)
            logits_sum = 0
            for model in models:
                logits_sum += model(imgs)
            preds = logits_sum.argmax(dim=1).cpu()
            preds_all.append(preds)
    return torch.cat(preds_all, dim=0)

# --------------------------- Main ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed_list', type=int, nargs='+', default=[42, 123, 2025])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transforms
    train_transform = transforms.Compose([
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.FashionMNIST(root=args.data_root, train=True, download=True, transform=train_transform)
    val_dataset = datasets.FashionMNIST(root=args.data_root, train=False, download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    all_models = []
    for seed in args.seed_list:
        print(f"\n=== Training model with seed {seed} ===")
        set_seed(seed)
        model = ResNetSE(num_classes=10).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr*4,
            total_steps=args.epochs*steps_per_epoch,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
        )

        best_acc = 0.0
        for epoch in range(1, args.epochs+1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer, scheduler,
                                                    mixup_alpha=0.4, cutmix_alpha=1.0)
            val_loss, val_acc = validate(model, device, val_loader, criterion)
            print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"resnetse_seed{seed}_best.pth")
        print(f"Best val acc for seed {seed}: {best_acc:.4f}")
        model.load_state_dict(torch.load(f"resnetse_seed{seed}_best.pth"))
        all_models.append(model)

    # Ensemble validation
    print("\n=== Ensemble Evaluation ===")
    preds = ensemble_predict(all_models, device, val_loader)
    labels = torch.tensor(val_dataset.targets)
    acc = (preds == labels).float().mean().item()
    print(f"Ensemble Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
