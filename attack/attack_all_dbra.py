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
import time
import numpy as np
from baseline.ITDS_main.defense_models.feature_squeezing import bit_depth_reduction
from baseline.ITDS_main.defense_models.JPEG import jpeg_compress
from baseline.ITDS_main.defense_models.median_smoothing import median_smoothing
from baseline.ITDS_main.defense_models.random_crop import random_crop_defense
CACHE_PATH = ''

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)



def tensor2img(images, path:str,labels=None, nrow=4, figsize=(8, 8), 
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
                    vic_nets:nn.Module,
                    malicious_score:torch.Tensor):
    adv_net.eval()
    
    unnormalizer = UnNormalize((0.4914, 0.4822, 0.4465),   # 3-channel mean
                                (0.2023, 0.1994, 0.2010))
    normalizer = Normalize((0.4914, 0.4822, 0.4465),   # 3-channel mean
                                (0.2023, 0.1994, 0.2010))
    ratio = dictionary_top_rate # select top features from trigger dictionary to add to input samples

    B,C,W,H = trigger_dictionary.shape

    # Use attribution for the specified label
    start_time = time.time()
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
    implanted_samples = unnormalizer(implanted_samples).clamp(0,1)
    implanted_samples = normalizer(implanted_samples)
    implanted_samples = bit_depth_reduction(implanted_samples)
    end_time = time.time()
    print('Runing Time:', end_time-start_time)

    attack_res = {}
    for i in range(len(vic_nets)):
        vic_net = vic_nets[i]
        vic_pred = torch.softmax(vic_net(implanted_samples),dim=-1)
        local_pred = torch.softmax(adv_net(implanted_samples),dim=-1)
        vic_suc = (vic_pred.argmax(-1)==tgt_label)
        local_suc = (local_pred.argmax(-1)==tgt_label)
        
        trans_suc_flag = vic_suc*local_suc
        if local_suc.sum().item() == 0:
            attack_res = {}
            for i in range(len(vic_nets)):
                attack_res[i] = (0, trans_suc_flag,local_suc)
            return attack_res
        
        overlapping_rate = (trans_suc_flag).sum().item()/local_suc.sum().item()

        attack_res[i] = overlapping_rate, trans_suc_flag, local_suc

    return attack_res

