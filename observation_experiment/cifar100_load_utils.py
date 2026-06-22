import torch
from net_structures.resnet_cifar100 import get_model as get_resnet_cifar100
from net_structures.vgg_cifar100 import get_model as get_vgg_cifar100



def load_resnet18():
    model = get_resnet_cifar100('resnet18',100)
    temp = torch.load('best_resnet18_cifar100.pth')
    model.load_state_dict(temp['model_state_dict'])
    return model


def load_resnet34():
    model = get_resnet_cifar100('resnet34',100)
    temp = torch.load('best_resnet34_cifar100.pth')
    model.load_state_dict(temp['model_state_dict'])
    return model


def load_vgg13_cifar100():
    model = get_vgg_cifar100('vgg13', num_classes=100)
    temp = (torch.load('best_vgg13_cifar100.pth'))
    model.load_state_dict(temp['model_state_dict'])
    return model


def load_vgg16_cifar100():
    model = get_vgg_cifar100('vgg16',num_classes=100)
    temp = (torch.load('best_vgg16_cifar100.pth'))
    model.load_state_dict(temp['model_state_dict'])
    return model

