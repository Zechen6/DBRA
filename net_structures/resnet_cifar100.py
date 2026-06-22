# ResNet34 and ResNet18 training for CIFAR100
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


# =========================================================
# 1. Set random seed
# =========================================================
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =========================================================
# 2. Build ResNet18 / ResNet34 for CIFAR100
# =========================================================
def get_model(model_name, num_classes=100, cifar_stem=True):
    model_name = model_name.lower()

    if model_name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # The original torchvision ResNet is for ImageNet 224x224 input: 7x7 stride=2 + maxpool
    # CIFAR100 is 32x32; typically change to 3x3 stride=1 and remove maxpool to preserve spatial information
    if cifar_stem:
        model.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =========================================================
# 3. Data loading: CIFAR100
# =========================================================
def get_dataloaders(data_root="./data", batch_size=128, num_workers=4):
    # Common mean/std for CIFAR100
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = torchvision.datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader


# =========================================================
# 4. Train one epoch
# =========================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, log_interval=100):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % log_interval == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# =========================================================
# 5. Evaluation
# =========================================================
@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# =========================================================
# 6. Main function
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--no_cifar_stem", action="store_true", help="Use torchvision original ImageNet stem; not recommended for CIFAR100")

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print(f"Model      : {args.model}")
    print("Dataset    : CIFAR100")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"Device     : {device}")
    print(f"CIFAR stem : {not args.no_cifar_stem}")
    print("=" * 80)

    train_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(
        args.model,
        num_classes=100,
        cifar_stem=not args.no_cifar_stem,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )

    best_acc = 0.0
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=args.log_interval,
        )

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Best Acc: {best_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model + early stopping counter
        if test_acc > best_acc + args.early_stop_min_delta:
            best_acc = test_acc
            best_epoch = epoch
            early_stop_counter = 0

            save_path = os.path.join(
                args.save_dir,
                f"best_{args.model}_cifar100.pth"
            )

            torch.save(
                {
                    "model": args.model,
                    "dataset": "CIFAR100",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_acc": best_acc,
                    "num_classes": 100,
                    "cifar_stem": not args.no_cifar_stem,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                save_path,
            )

            print(f"Saved best model to: {save_path}")
        else:
            early_stop_counter += 1
            print(
                f"EarlyStopping counter: "
                f"{early_stop_counter}/{args.early_stop_patience}"
            )

            if early_stop_counter >= args.early_stop_patience:
                print("=" * 80)
                print(f"Early stopping triggered: Test Acc did not improve for {args.early_stop_patience} consecutive epochs")
                print(f"Stopped at epoch: {epoch}")
                print(f"Best Epoch: {best_epoch}")
                print(f"Best Test Acc: {best_acc:.2f}%")
                print("=" * 80)
                break

    print("=" * 80)
    print(f"Training completed: {args.model}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Test Acc: {best_acc:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
