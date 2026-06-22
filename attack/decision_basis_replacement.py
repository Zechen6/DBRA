import sys
cwd = '/'
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
CACHE_PATH = 'cache/cifar10/'

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
    

def fed_ffc(global_model:nn.Module, 
            local_model:nn.Module,
            sample:torch.Tensor,
            label:torch.Tensor,
            ffc_params:dict=trigger_extraction_ffc_params):
    lr = ffc_params['lr']
    echo = ffc_params['echo']
    y = label*torch.ones(sample.shape[0]).to(device).long()
    global_model.eval()
    local_model.eval()
    with torch.enable_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=lr)
        optimizer_l = torch.optim.SGD(local_model.parameters(), lr=lr)
        data_new_g = sample.clone()
        data_new_l = sample.clone()
        data_new_g.requires_grad = True
        data_new_l.requires_grad = True
        for e in range(echo):
            pred_g = global_model(data_new_g)
            pred_l = local_model(data_new_l)
            loss_g = loss_fn(pred_g, y)
            loss_l = loss_fn(pred_l, y)
            optimizer_g.zero_grad()
            optimizer_l.zero_grad()
            loss_g.backward()
            loss_l.backward()
            grad_g = data_new_g.grad.data.clone()        
            grad_l = data_new_l.grad.data.clone()
            with torch.no_grad():
                data_new_g -= lr*grad_g
                data_new_g.grad.zero_()
                data_new_l -= lr*grad_l
                data_new_l.grad.zero_()
        ori_freq = torch.fft.fft2(sample)
        new_freq_g = torch.fft.fft2(data_new_g)
        new_freq_l = torch.fft.fft2(data_new_l)

    if target_label in [0, 3]: # These are the main classes for client 5
        global_local_mutual_energy = 2*((new_freq_l-new_freq_g)*ori_freq.conj()).real
        proj_gl_me = global_local_mutual_energy/((new_freq_l-new_freq_g).abs()+1e-7)
    else:
        global_local_mutual_energy = 2*((new_freq_g-new_freq_l)*ori_freq.conj()).real
        proj_gl_me = global_local_mutual_energy/((new_freq_g-new_freq_l).abs()+1e-7)

    global_local_both_enhanced = 0.5*(new_freq_g+new_freq_l)
    signal_both_ori_freq_me = 2*(global_local_both_enhanced*ori_freq.conj()).real
    signal_both_proj = signal_both_ori_freq_me/(global_local_both_enhanced.abs()+1e-7)
    signal_diff_score \
        = signal_both_proj - proj_gl_me
    
    
    return global_local_mutual_energy, signal_diff_score


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
    unnormalizer = UnNormalize((0.4914, 0.4822, 0.4465),   # 3-channel mean
                                (0.2023, 0.1994, 0.2010))
    normalizer = Normalize((0.4914, 0.4822, 0.4465),   # 3-channel mean
                                (0.2023, 0.1994, 0.2010))
    adv_net.eval()
    vic_net.eval()
    max_len = 30
    interval = 3
    suc_interval_list = [0 for i in range(max_len // interval)]
    suc_accumulate_list = [0 for i in range(max_len // interval)]
    ratio = dictionary_top_rate 

    B,C,W,H = trigger_dictionary.shape

    # Use attribution for the specified label

    mag_score, feature_mag = find_most_important_fea_by_mag(vic_sample, adv_net,None)

    transfer_score = mag_score
    tgt_freqs = torch.fft.fft2(trigger_dictionary)
    ba_freqs = torch.fft.fft2(feature_mag)
    ba_masks = torch.where(ba_freqs.abs()>1e-5, 1, 0)

    
    ba_ratio = ba_masks.sum()/(3*32*32)

    if ba_ratio > ratio:
        print("Exceeded")
        ba_masks = select_top_element(transfer_score, ratio)
    ba_ratio = ba_masks.sum()/(3*32*32)

    mask4adv_feature = select_top_element(malicious_score, dictionary_top_rate)

    vic_sample_freqs = torch.fft.fft2(vic_sample)

    
    selected_adv_features_mask = (mask4adv_feature+ba_masks)

    implanted_samples = torch.fft.ifft2(implanted_signal_rate*tgt_freqs*selected_adv_features_mask+\
                                    (vic_sample_freqs*(1-ba_masks))).real
    
    implanted_samples = unnormalizer(implanted_samples).clamp(0,1)
    implanted_samples = normalizer(implanted_samples)
    
    
    vic_pred = torch.softmax(vic_net(implanted_samples),dim=-1)
    local_pred = torch.softmax(adv_net(implanted_samples),dim=-1)
    vic_suc = (vic_pred.argmax(-1)==tgt_label)
    local_suc = (local_pred.argmax(-1)==tgt_label)
    
    trans_suc_flag = vic_suc*local_suc
    if local_suc.sum().item() == 0:
        return suc_accumulate_list, suc_interval_list, -1, 0, local_suc
    
    overlapping_rate = (trans_suc_flag).sum().item()/local_suc.sum().item()
    suc_flag = 0
    selected_conf_samples = None
    if overlapping_rate > 0:
        
        target_conf = local_pred[:,target_label]
        top_k = min(max_len, len(target_conf))
        topk_conf_v, topk_conf_i = torch.topk(target_conf, k=top_k)
        selected_conf_samples = implanted_samples[topk_conf_i]
        conf_preds = vic_net(selected_conf_samples)
        suc_flag = (conf_preds.argmax(-1)==tgt_label)
        step = top_k // interval
        for i in range(1,step+1):
            if suc_flag[:i*interval].sum().item() > 0:
                suc_accumulate_list[i-1] += 1
                suc_interval_list[i-1] += 1 if suc_flag[(i-1)*interval:i*interval].sum() > 0 else 0

    return suc_accumulate_list, suc_interval_list, 0, 0, local_suc

