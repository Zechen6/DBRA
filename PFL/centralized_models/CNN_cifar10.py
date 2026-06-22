#Train CIFAR-10 classification models with CNN5 or CNN7.
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#CNN5 model
class CNN5(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN5, self).__init__()

        self.features = nn.Sequential(
            # Conv 1: 3 x 32 x 32 -> 32 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  

            # Conv 2: 32 x 16 x 16 -> 64 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   

            # Conv 3: 64 x 8 x 8 -> 128 x 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 128 x 4 x 4
        )

        self.classifier = nn.Sequential(
            # FC 1
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # FC 2
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


#CNN7 model
class CNN7(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN7, self).__init__()

        self.features = nn.Sequential(
            # Conv 1: 3 x 32 x 32 -> 32 x 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Conv 2: 32 x 32 x 32 -> 64 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   

            # Conv 3: 64 x 16 x 16 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   

            # Conv 4: 128 x 8 x 8 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 256 x 4 x 4
        )

        self.classifier = nn.Sequential(
            # FC 1
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # FC 2
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # FC 3
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def get_model(model_name, num_classes=10):
    model_name = model_name.lower()

    if model_name == "cnn5":
        return CNN5(num_classes=num_classes)
    elif model_name == "cnn7":
        return CNN7(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")


def get_dataloaders(data_root="/home/zyq/PFLlib/system/data", batch_size=256, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010),
        ),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    print("CIFAR10 data root:", data_root)
    print("Train size:", len(train_set))
    print("Test size :", len(test_set))

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    return avg_loss, acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total

    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="cnn5", choices=["cnn5", "cnn7"])
    parser.add_argument("--data_root", type=str, default="/home/zyq/PFLlib/system/data")
    parser.add_argument("--save_dir", type=str, default="/home/zyq/PFLlib/system/cnn_cifar10")

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)

    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print(f"Model      : {args.model}")
    print(f"Dataset    : CIFAR10")
    print(f"Data root  : {args.data_root}")
    print(f"Save dir   : {args.save_dir}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"Device     : {device}")
    print(f"AMP        : {device.type == 'cuda'}")
    print("=" * 80)

    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(args.model, num_classes=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc = 0.0
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Best Acc: {best_acc:.2f}%"
        )

        if test_acc > best_acc + args.early_stop_min_delta:
            best_acc = test_acc
            early_stop_counter = 0

            save_path = os.path.join(
                args.save_dir,
                f"best_{args.model}_cifar10.pth"
            )

            torch.save(
                {
                    "model": args.model,
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )

        else:
            early_stop_counter += 1
            print(
                f"EarlyStopping counter: "
                f"{early_stop_counter}/{args.early_stop_patience}"
            )

            if early_stop_counter >= args.early_stop_patience:
                print("=" * 80)
                print(f"Best Test Acc: {best_acc:.2f}%")
                print("=" * 80)
                break

    print("=" * 80)
    print(f"Best Test Acc: {best_acc:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()