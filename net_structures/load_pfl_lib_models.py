import torch
import torchvision
import copy
import torch.nn as nn
from net_structures.compatible_file import BaseHeadSplit

def load_fedrep_cifar10_models(model_name:str, client_id:int, model_path:str, model=None):
    """
    Load FedRep model
    """
    model_path += f'{client_id}/best_model.pt'
    model_infos = torch.load(model_path)
    if model_name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=10)
        head = copy.deepcopy(net.fc)
        net.fc = nn.Identity()
        net = BaseHeadSplit(net, head)
    if model_name == None or model_name == '':
        net = model
        if net == None:
            raise ValueError('model_name is None or empty, and model is also None')
    net.load_state_dict(model_infos['model_state_dict'])
    return net, model_infos


def load_non_operate_cifar10_models(model_name:str, client_id:int, model_path:str, model=None):
    """
    Load Ditto model
    """
    model_path += f'{client_id}/best_model.pt'
    model_infos = torch.load(model_path)
    if model_name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if model_name == None or model_name == '':
        net = model
        if net == None:
            raise ValueError('model_name is None or empty, and model is also None')
    net.load_state_dict(model_infos['model_state_dict'])
    return net, model_infos


def load_fedala_cifar10_models(model_name:str, client_id:int, model_path:str, model=None):
    """
    Load FedALA model
    """
    model_path += f'{client_id}/best_model.pt'
    model_infos = torch.load(model_path)
    if model_name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if model_name == None or model_name == '':
        net = model
        if net == None:
            raise ValueError('model_name is None or empty, and model is also None')
    net.load_state_dict(model_infos['model_state_dict'])
    return net, model_infos


def load_non_operate_svhn_models(model_name:str, client_id:int, model_path:str, model=None):
    """
    Load PFLLib models that do not require reprocessing
    """
    model_path += f'{client_id}/best_model.pt'
    model_infos = torch.load(model_path)
    if model_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=False, num_classes=10)
    if model_name == None or model_name == '':
        net = model
        if net == None:
            raise ValueError('model_name is None or empty, and model is also None')
    net.load_state_dict(model_infos['model_state_dict'])
    return net, model_infos


def load_fedrep_svhn_models(model_name:str, client_id:int, model_path:str, model=None):
    """
    Load PFLLib models that do not require reprocessing
    """
    model_path += f'{client_id}/best_model.pt'
    model_infos = torch.load(model_path)
    if model_name == 'resnet50':
        net = torchvision.models.resnet50(pretrained=False, num_classes=10)
        head = copy.deepcopy(net.fc)
        net.fc = nn.Identity()
        net = BaseHeadSplit(net, head)
    if model_name == None or model_name == '':
        net = model
        if net == None:
            raise ValueError('model_name is None or empty, and model is also None')
    net.load_state_dict(model_infos['model_state_dict'])
    return net, model_infos
