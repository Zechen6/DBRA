import sys
cwd = '../'
sys.path.insert(0, cwd)

from confs.device_conf import device
from confs.implantation_confs import *
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from attack.attribution_methods.ffc import find_most_important_feature, select_bottom_element, select_top_element
from attack.attribution_methods.ffc import malicious_ffc, ffc
from attack.attribution_methods.ffc import find_most_important_feature, find_top_malicious_feature
from attack.attribution_methods.ffc import find_most_important_fea_by_mag
from confs.data_conf import cifar10_root
import matplotlib.pyplot as plt
import numpy as np
CACHE_PATH = ''

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def tensor2img(images, path:str,labels=None, nrow=1, figsize=(8, 8), 
               unnormalize=True, 
               mean=(0.4914, 0.4822, 0.4465), 
               std=(0.2023, 0.1994, 0.2010)):
    """
    Visualize CIFAR-10 images
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010)
    Args:
        images: Tensor or ndarray, shape (B, C, H, W) or (C, H, W)
        labels: list or Tensor, optional
        nrow: number of images per row
        figsize: figure size
        unnormalize: whether to unnormalize (if you applied normalization)
        mean, std: used for unnormalization
    """
    
    # Convert to batch form
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    
    if images.ndim == 3:
        images = images.unsqueeze(0)

    B = images.shape[0]

    # Unnormalize (if needed)
    if unnormalize and mean is not None and std is not None:
        mean = torch.tensor(mean).view(1, -1, 1, 1)
        std = torch.tensor(std).view(1, -1, 1, 1)
        images = images * std + mean

    images = images.numpy()

    ncol = (B + nrow - 1) // nrow

    plt.figure(figsize=figsize)

    for i in range(B):
        plt.subplot(ncol, nrow, i + 1)
        
        img = images[i].transpose(1, 2, 0)  # CHW -> HWC
        
        # Clip to prevent display issues
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.axis('off')

        if labels is not None:
            label = labels[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            plt.title(CIFAR10_CLASSES[label], fontsize=8)

    plt.tight_layout()
    plt.savefig(path)


def refined_trigger_implantation(vic_sample:torch.Tensor,
                    adv_net:nn.Module,
                    tgt_label:torch.Tensor|int,
                    trigger_dictionary:torch.Tensor,
                    vic_net:nn.Module,
                    malicious_score:torch.Tensor,
                    original_triggers:torch.Tensor=None):
    adv_net.eval()
    vic_net.eval()

    ratio = dictionary_top_rate # select features from the trigger dictionary to add to input samples

    mag_score, feature_mag = find_most_important_fea_by_mag(vic_sample, adv_net,None)

    transfer_score = mag_score
    tgt_freqs = torch.fft.fft2(trigger_dictionary)
    ba_freqs = torch.fft.fft2(feature_mag)
    ba_masks = torch.where(ba_freqs.abs()>1e-5, 1, 0)


    ba_ratio = ba_masks.sum()/(3*32*32)

    if ba_ratio > ratio:
        ba_masks = select_top_element(transfer_score, ratio)
    ba_ratio = ba_masks.sum()/(3*32*32)

    mask4adv_feature = select_top_element(malicious_score, dictionary_top_rate)

    vic_sample_freqs = torch.fft.fft2(vic_sample)

    
    selected_adv_features_mask = (mask4adv_feature+ba_masks)

    implanted_samples = torch.fft.ifft2(implanted_signal_rate*tgt_freqs*selected_adv_features_mask+\
                                    (vic_sample_freqs*(1-ba_masks))).real
    

    vic_pred = torch.softmax(vic_net(implanted_samples),dim=-1)
    local_pred = torch.softmax(adv_net(implanted_samples),dim=-1)
    vic_suc = (vic_pred.argmax(-1)==tgt_label)
    local_suc = (local_pred.argmax(-1)==tgt_label)
    
    trans_suc_flag = vic_suc*local_suc
    if local_suc.sum().item() == 0:

        return 0, trans_suc_flag, -1, 0, local_suc
    
    overlapping_rate = (trans_suc_flag).sum().item()/local_suc.sum().item()



    return overlapping_rate, trans_suc_flag, local_suc

