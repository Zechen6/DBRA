import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from collections import defaultdict

MINI_IMGNET_ROOT = ''

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False


from torch.utils.data import Dataset

class TransformSubset(Dataset):

    def __init__(self, subset, transform=None):
        self.subset = subset      
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx] 
        if self.transform:
            x = self.transform(x)
        return x, y


class MiniImageNetDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root: Dataset Path for example: dataset/miniImageNet
        transform: torchvision transforms
        """
        self.root = root
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

        # class_name -> label index
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}


        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def load_mini_imgnet_src(batch_size:128, shuffle=False):

    transform = transforms.Compose([
    transforms.Resize(256),          
    transforms.CenterCrop(224),      
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])
    dataset = ImageFolder(root="dataset/miniImageNet", transform=transform)
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=4,
    pin_memory=True
    )
    
    return dataloader

def stratified_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
    verbose: bool = True
    ):

    # Check whether the dataset has a 'samples' attribute (ImageFolder provides this)
    if not hasattr(dataset, 'samples'):
        raise AttributeError("Dataset is missing 'samples' attribute; cannot retrieve class information. "
                             "Please use torchvision.datasets.ImageFolder or a similar dataset.")
    
    # 1. Get index lists for each class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    # 2. Split each class separately
    for label, indices in class_indices.items():
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        if n_train == 0:
            # If a class has too few samples, ensure at least one sample in the training set
            n_train = 1
        if n_train >= n_total:
            n_train = n_total - 1  # Ensure at least one sample remains for validation
        
        # Shuffle the indices for this class with a fixed seed
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(indices), generator=rng).tolist()

        # Split according to the ratio
        train_idx = [indices[i] for i in perm[:n_train]]
        val_idx = [indices[i] for i in perm[n_train:]]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        
        if verbose:
            print(f"Class {label}: total {n_total} | train {len(train_idx)} | val {len(val_idx)}")
    
    # Optionally shuffle the overall train and validation index order (random but repeatable)
    rng_train = torch.Generator().manual_seed(seed + 1)
    rng_val = torch.Generator().manual_seed(seed + 2)
    train_indices = [train_indices[i] for i in torch.randperm(len(train_indices), generator=rng_train).tolist()]
    val_indices = [val_indices[i] for i in torch.randperm(len(val_indices), generator=rng_val).tolist()]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    if verbose:
        print(f"\nDataset split complete: train {len(train_dataset)} images, val {len(val_dataset)} images")
    
    return train_dataset, val_dataset


def load_split_dataset(batch_size):
    set_seed(1234)
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),        # Randomly crop and resize to 224
    transforms.RandomHorizontalFlip(p=0.5),   # Random horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random brightness/contrast
    transforms.ToTensor(),                    # Convert to Tensor (0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalize
    ])
    val_transform = transforms.Compose([
    transforms.Resize(256),          # Resize first
    transforms.CenterCrop(224),      # Then center crop
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ])

    # 2. Use ImageFolder to load data
    full_dataset = ImageFolder(root=MINI_IMGNET_ROOT, transform=None)
    print(f"Dataset contains {len(full_dataset)} images")
    print(f"Dataset classes: {full_dataset.classes}")

    # 3. Split train set and validation set (8:2)
    train_subset, val_subset = stratified_split(full_dataset,0.9,1234,True)
    train_dataset = TransformSubset(train_subset, transform=train_transform)
    val_dataset = TransformSubset(val_subset, transform=val_transform)
    # 4. Create DataLoader

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

