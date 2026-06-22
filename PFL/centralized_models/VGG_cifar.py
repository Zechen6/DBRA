#Train CIFAR-100/CIFAR-10 classification models with VGG16 or VGG13.
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


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(model_name, num_classes=10):
    model_name = model_name.lower()

    if model_name == "vgg13":
        model = models.vgg13(weights=None)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model


def get_dataloaders(dataset="cifar100", data_root="./data", batch_size=128, num_workers=4):
    dataset = dataset.lower()

    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        dataset_cls = torchvision.datasets.CIFAR10
    elif dataset == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        dataset_cls = torchvision.datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported Dataset: {dataset}")

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

    train_set = dataset_cls(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = dataset_cls(
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
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

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

    parser.add_argument("--model", type=str, default="vgg13", choices=["vgg13", "vgg16"])
    parser.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=20)
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
    print(f"Dataset    : {args.dataset.upper()}")
    print(f"Epochs     : {args.epochs}")
    print(f"Batch size : {args.batch_size}")
    print(f"LR         : {args.lr}")
    print(f"Device     : {device}")
    print("=" * 80)

    train_loader, test_loader = get_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = 100 if args.dataset.lower() == "cifar100" else 10
    model = get_model(args.model, num_classes=num_classes)
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
                f"best_{args.model}_{args.dataset.lower()}.pth"
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

            print(f"Save the best model to: {save_path}")
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