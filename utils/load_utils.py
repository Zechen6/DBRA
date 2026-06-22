import sys
cwd = '../'
sys.path.insert(0, cwd)

from confs.device_conf import device
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from attack.attribution_methods.ffc import find_most_important_feature
from confs.data_conf import cifar10_root
from net_structures.load_pfl_lib_models import load_fedrep_cifar10_models, load_fedrep_svhn_models
from net_structures.load_pfl_lib_models import load_non_operate_cifar10_models
from net_structures.load_pfl_lib_models import load_fedala_cifar10_models

from net_structures.load_pfl_lib_models import load_non_operate_svhn_models
from utils.pfl_dataset_utils import load_train_data, load_test_data
from confs.implantation_confs import *


import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.io as sio
import random as rd

rd.seed(1234)


MODEL_PATH_PREFIX = ''

# ----- FEDREP -----
FEDREP_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedRep-Cifar10-ResNet18/client_'
FEDREP_MODEL_SVHN_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedRep-SVHN-ResNet50/client_'


# ----- DITTO -----
DITTO_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + '/Ditto-Cifar10-ResNet18/client_'
DITTO_MODEL_SVHN_DIR_PATH \
    = MODEL_PATH_PREFIX + '/Ditto-SVHN-ResNet50/client_'

# ----- FEDDBE -----
FEDDBE_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedDBE-Cifar10-ResNet18/client_'

# ----- FEDBN -----
FEDBN_MODEL_SVHN_DIR_PATH \
    = MODEL_PATH_PREFIX + 'FedBN-SVHN-ResNet50/client_'
FEDBN_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + 'FedBN-Cifar10-ResNet18/client_'

# ----- FEDALA -----
FEDALA_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedALA-Cifar10-ResNet18/client_'
FEDALA_MODEL_SVHN_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedALA-SVHN-ResNet50/client_'

# ----- FEDCAC -----
FEDCAC_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedCAC-Cifar10-ResNet18/client_'
FEDCAC_MODEL_SVHN_DIR_PATH \
    = MODEL_PATH_PREFIX + '/FedCAC-SVHN-ResNet50/client_'

# ----- SCAFFOLD -----
SCAFFOLD_MODEL_CIFAR10_DIR_PATH \
    = MODEL_PATH_PREFIX + '/SCAFFOLD-Cifar10-ResNet18/client_'
SCAFFOLD_MODEL_SVHN_DIR_PATH \
    = MODEL_PATH_PREFIX + '/SCAFFOLD-SVHN-ResNet50/client_'


CIFAR_CLASSNUM = 10

class ListDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x, y = self.data_list[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def load_clean_cifar10(batch_size=128, shuffle=True, data_aug=True):
    if data_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),   # 四周补 4 像素后随机裁剪
            transforms.RandomHorizontalFlip(),   
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),   # 三通道均值
                                (0.2023, 0.1994, 0.2010)),  # 三通道方差
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),   # 三通道均值
                                (0.2023, 0.1994, 0.2010)),  # 三通道方差
        ])


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


    train_dataset = datasets.CIFAR10(
        root=cifar10_root,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=cifar10_root,
        train=False,
        download=True,
        transform=test_transform
    )

    # --- DataLoader ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

    return train_loader, test_loader


def load_val_svhn(batch_size=128, shuffle=True, rand_sample=True):
    dataset_image = []
    dataset_label = []
        
    # Load SVHN data

    f='SVHN_Cropped/'
    train = sio.loadmat(f + "/train_32x32.mat")
    test = sio.loadmat(f + '/test_32x32.mat')


    test_data = test['X']     # (32, 32, 3, 26032)

    test_labels = test['y'].flatten()


    test_data = np.transpose(test_data, (3, 2, 0, 1))


    all_data = np.concatenate([test_data], axis=0)      # (99289, 32, 32, 3)
    all_labels = np.concatenate([test_labels], axis=0) # (99289,)


    all_labels[all_labels == 10] = 0
    dataset_image = all_data
    dataset_label = all_labels
    X = torch.Tensor(dataset_image).type(torch.float32)
    Y = torch.Tensor(dataset_label).type(torch.int64)

    data_list = [(x,y) for x, y in zip(X,Y)]

    list_dataset = ListDataset(data_list, None)
    data_loader = DataLoader(list_dataset, shuffle=False, batch_size=batch_size)
    return data_loader


def load_cifar10_sources(fed_name:str, model_name:str, victim_id=victim_client):

    if fed_name == 'FedRep':
        if model_name.lower() == 'resnet18':
            adv_model, _ = load_fedrep_cifar10_models(model_name, adv_client, 
                                                      FEDREP_MODEL_CIFAR10_DIR_PATH)
            victim_model, _ = load_fedrep_cifar10_models(model_name, victim_id, 
                                                         FEDREP_MODEL_CIFAR10_DIR_PATH)
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')
    elif fed_name.lower() == 'ditto':
        if model_name.lower() == 'resnet18':
            adv_model, _ \
                = load_non_operate_cifar10_models(model_name, adv_client, 
                                            DITTO_MODEL_CIFAR10_DIR_PATH, 
                                            )
            victim_model, _ \
                = load_non_operate_cifar10_models(model_name, victim_id, 
                                            DITTO_MODEL_CIFAR10_DIR_PATH, )
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')

    elif fed_name.lower() == 'fedala':
        if model_name.lower() == 'resnet18':
            adv_model, _ \
                = load_fedala_cifar10_models(model_name, adv_client, 
                                            FEDALA_MODEL_CIFAR10_DIR_PATH, 
                                            )
            victim_model, _ \
                = load_fedala_cifar10_models(model_name, victim_id, 
                                            FEDALA_MODEL_CIFAR10_DIR_PATH, )
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')
    elif fed_name.lower() in ['fedcac']:
        if model_name.lower() == 'resnet18':
            adv_model, _ \
                = load_non_operate_cifar10_models(model_name, adv_client, 
                                            FEDCAC_MODEL_CIFAR10_DIR_PATH, 
                                            )
            victim_model, _ \
                = load_non_operate_cifar10_models(model_name, victim_id, 
                                            FEDCAC_MODEL_CIFAR10_DIR_PATH, )
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')
    elif fed_name.lower() in ['scaffold']:
        if model_name.lower() == 'resnet18':
            adv_model, _ \
                = load_non_operate_cifar10_models(model_name, adv_client, 
                                            SCAFFOLD_MODEL_CIFAR10_DIR_PATH, 
                                            )
            victim_model, _ \
                = load_non_operate_cifar10_models(model_name, victim_id, 
                                            SCAFFOLD_MODEL_CIFAR10_DIR_PATH, )
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')
    else:
        raise NotImplementedError(f'{fed_name} is not supported for now.')

    
    train_loader, test_loader = load_clean_cifar10(batch_size=128, shuffle=False, data_aug=False)
    client_test_loader = load_test_data('Cifar10', adv_client, batch_size=128)
    client_train_loader = load_train_data('Cifar10', adv_client, batch_size=128)
    return train_loader, test_loader, client_train_loader, client_test_loader, adv_model, victim_model


def load_svhn_sources(fed_name:str, model_name:str, victim_id=victim_client):
    
    if fed_name.lower() in ['fedbn','ditto','scaffold','fedcac','fedala']:
        dir_path = None
        if fed_name.lower() == 'fedrep':
            dir_path = FEDBN_MODEL_SVHN_DIR_PATH
        elif fed_name.lower() == 'ditto':
            dir_path = DITTO_MODEL_SVHN_DIR_PATH
        elif fed_name.lower() == 'scaffold':
            dir_path = SCAFFOLD_MODEL_SVHN_DIR_PATH
        elif fed_name.lower() == 'fedcac':
            dir_path = FEDCAC_MODEL_SVHN_DIR_PATH
        elif fed_name.lower() == 'fedala':
            dir_path = FEDALA_MODEL_SVHN_DIR_PATH
        else:
            raise NotImplementedError(f'{fed_name} is not supported for now.')
        if model_name.lower() == 'resnet50':
            adv_model, _ \
                = load_non_operate_svhn_models(model_name, adv_client, 
                                            dir_path, )
            victim_model, _ \
                = load_non_operate_svhn_models(model_name, victim_id, 
                                            dir_path, )
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')
    elif fed_name.lower() == 'fedrep':
        if model_name.lower() == 'resnet50':
            adv_model, _ \
                = load_fedrep_svhn_models(model_name, adv_client, 
                                            FEDREP_MODEL_SVHN_DIR_PATH, 
                                            )
            victim_model, _ \
                = load_fedrep_svhn_models(model_name, victim_id, 
                                            FEDREP_MODEL_SVHN_DIR_PATH, )
            adv_model.to(device)
            victim_model.to(device)
        else:
            raise NotImplementedError(f'{model_name} is not supported for now.')
    else:
        raise NotImplementedError(f'{fed_name} is not supported for now.')

    train_loader = None
    test_loader = load_val_svhn(batch_size=128, shuffle=False, rand_sample=False)

    client_test_loader = load_test_data('SVHN', adv_client, batch_size=128)
    client_train_loader = load_train_data('SVHN', adv_client, batch_size=128)
    return train_loader, test_loader, client_train_loader, client_test_loader, adv_model, victim_model
