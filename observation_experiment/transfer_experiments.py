
import sys
cwd = '../'
sys.path.insert(0, cwd)

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from confs.device_conf import device
from attack.attribution_methods.ffc import find_most_important_fea_by_mag, find_most_important_feature
from attack.attribution_methods.ffc import select_bottom_element, select_top_element
from observation_experiment.mini_imgnet_load_utils import load_mini_imgnet_src, load_split_dataset

TGT_LABEL = 10
ffc_settings = {'lr':1000, 'echo':10}
FIG_EN = False
MINI_CLASS_NUM = 100

def load_vit_b_32():
    net = models.vit_b_32()
    net.heads.head = nn.Linear(net.heads.head.in_features, MINI_CLASS_NUM) 
    net.load_state_dict(torch.load('observation_experiment/models/ViT-b-32-weights.pt'))
    net.eval()
    return net


def load_vit_b_16():
    net = models.vit_b_16()
    net.heads.head = nn.Linear(net.heads.head.in_features, MINI_CLASS_NUM) 
    net.load_state_dict(torch.load('observation_experiment/models/ViT-b-16-weights.pt'))
    net.eval()
    return net


def load_resnet50():
    net = models.resnet50()
    net.fc = nn.Linear(net.fc.in_features, MINI_CLASS_NUM)
    net.load_state_dict(torch.load('observation_experiment/models/Resnet50-weights.pt'))
    net.eval()
    return net


def load_resnet101():
    net = models.resnet101()
    net.fc = nn.Linear(net.fc.in_features, MINI_CLASS_NUM)
    net.load_state_dict(torch.load('observation_experiment/models/Resnet101-weights.pt'))
    net.eval()
    return net


def load_resnet152():
    net = models.resnet152()
    net.fc = nn.Linear(net.fc.in_features, MINI_CLASS_NUM)
    net.load_state_dict(torch.load('observation_experiment/models/Resnet152-weights.pt'))
    net.eval()
    return net


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
        
        # clip to prevent display issues
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.axis('off')

        if labels is not None:
            label = labels[i]
            if isinstance(label, torch.Tensor):
                label = label.item()

    plt.tight_layout()
    plt.savefig(path)


def decision_basis_replacement(vic_sample:torch.Tensor,
                    adv_net:nn.Module,
                    tgt_label:torch.Tensor|int,
                    trigger_dictionary:torch.Tensor,
                    vic_net:nn.Module,
                    malicious_score:torch.Tensor):
    adv_net.eval()
    vic_net.eval()
    
    ratio = 0.9 # select top features from trigger dictionary to add to input samples
    top_bone_ratio=0.0003
    implanted_trigger_mag = 2
    B,C,W,H = trigger_dictionary.shape
        
    # Use attribution for the specified label
    
    mag_score, feature_mag = find_most_important_feature(vic_sample, adv_net, ffc_settings)

    recurrent_adv_score, feature_malicious = find_most_important_feature(vic_sample, adv_net, ffc_settings)
    for _ in range(1):
        recurrent_adv_score, feature_malicious = find_most_important_feature(feature_malicious, adv_net,ffc_settings)

    transfer_score = mag_score
    tgt_freqs = torch.fft.fft2(trigger_dictionary)

    top_bone_mask = select_top_element(recurrent_adv_score, top_bone_ratio)

    ba_shape_mask = 1-select_top_element(transfer_score, ratio)

    adv_masks = select_top_element(malicious_score, top_bone_ratio)
    vic_sample_freqs = torch.fft.fft2(vic_sample)

    
    selected_adv_features_mask = torch.where(adv_masks+top_bone_mask>0,1,0)

    implanted_samples = torch.fft.ifft2(implanted_trigger_mag*tgt_freqs*selected_adv_features_mask+\
                                      (vic_sample_freqs*(1-selected_adv_features_mask-ba_shape_mask))).real
    

    vic_pred = torch.softmax(vic_net(implanted_samples),dim=-1)
    local_pred = torch.softmax(adv_net(implanted_samples),dim=-1)
    vic_suc = (vic_pred.argmax(-1)==tgt_label)
    local_suc = (local_pred.argmax(-1)==tgt_label)
    
    trans_suc_flag = vic_suc*local_suc
    if local_suc.sum().item() == 0:
        return 0, False, local_suc
    
    overlapping_rate = (trans_suc_flag).sum().item()/local_suc.sum().item()
    suc_flag = 0

    if overlapping_rate > 0:
        selected_trigger_patterns = implanted_trigger_mag*tgt_freqs*selected_adv_features_mask


    
    if trans_suc_flag.sum() > 0 and FIG_EN:
        temp = (implanted_samples[trans_suc_flag])
        pattern_temp = torch.fft.ifft2(selected_trigger_patterns[trans_suc_flag]).real
        if len(temp) > 1:
            idx = np.random.randint(0,len(temp)-1)
        else:
            idx = 0
        for idx in range(len(temp)):
            success_idx = torch.nonzero(trans_suc_flag)
            print(idx)
            selected_original_idx = success_idx[idx]
            decision_basis = torch.fft.ifft2(vic_sample_freqs*selected_adv_features_mask[selected_original_idx]).real
            detail_signals = torch.fft.ifft2(vic_sample_freqs*ba_shape_mask).real
            rest_sample = torch.fft.ifft2(vic_sample_freqs*(1-selected_adv_features_mask-ba_shape_mask)).real
            imgs = torch.stack([
                rest_sample[idx],
                detail_signals[0],
                decision_basis[0],
                trigger_dictionary[selected_original_idx[0]],
                vic_sample[0],
                temp[idx],
                pattern_temp[idx]],dim=0)
            
            tensor2img(imgs,
                    path=f'decision-basis.png')
            imgs = torch.stack([
                vic_sample[0],
                temp[idx],
                pattern_temp[idx]],dim=0)
            
            tensor2img(imgs,
                    path=f'modified-feature.png')
            
            print('drawn')

    return suc_flag, trans_suc_flag,  local_suc


def get_src_net_trigger_feature(net_src:nn.Module, 
                                net_structure:str):

    train_loader = load_mini_imgnet_src(32)
    net_src.to(device)
    net_src.eval()
    # First extract features for the tgt_label
    trigger_features = []
    trigger_scores = []
    for batch_idx, (X,y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        pred_src = net_src(X).argmax(-1)
        print(pred_src)
        X = X[pred_src==TGT_LABEL]
        if len(X) <= 0:
            continue
        scores_origin, features = find_most_important_feature(X, net_src, ffc_settings)
        for _ in range(2):
            scores_new, features = find_most_important_feature(features, net_src, ffc_settings)
        
        trigger_features.append(features)
        trigger_scores.append(scores_new)
    trigger_features = torch.cat(trigger_features,dim=0)
    trigger_scores = torch.cat(trigger_scores, dim=0)
    res = {'features':trigger_features,'scores':trigger_scores}
    torch.save(res,f'observation_experiment/{net_structure}-trigger_infos.pt')


def test_transfer(net_src:nn.Module, 
                  net_tgt:nn.Module,
                  net_structure_src:str,
                  net_structure_tgt:str):
    net_src.to(device)
    net_tgt.to(device)
    net_src.eval()
    net_tgt.eval()
    batch_size = 128
    train_loader, val_loader = load_split_dataset(batch_size)
    trigger_info = torch.load(f'observation_experiment/{net_structure_src}-trigger_infos.pt')
    trigger_features = trigger_info['features']
    trigger_scores =  trigger_info['scores']
    transfer_success_cn = 0
    cn = 0
    for batch_idx, (X,y) in enumerate(val_loader):

        X, y = X.to(device), y.to(device)
        pred_tgt = net_tgt(X).argmax(-1)
        X = X[pred_tgt!=TGT_LABEL]
        ori_y = y[pred_tgt!=TGT_LABEL]
        if len(X) <=0:
            continue
        for i in range(X.shape[0]):
            cn += 1
            print(ori_y[i])
            _,transfer_success,local_suc_flag \
                = decision_basis_replacement(X[i].unsqueeze(0), net_src, TGT_LABEL, 
                                            trigger_features, net_tgt,
                                            trigger_scores)
            transfer_success_cn += 1 if transfer_success >0 else 0

            print(f'Sample {cn} processed.')
            print(f'Success triggers found: {transfer_success_cn}/{cn}={100*transfer_success_cn/cn:.4f}%')
            
    with open(f'observation_experiment/{net_structure_src}-{net_structure_tgt}_transfer.res', 'a') as f:
        print(f'Success triggers found: {transfer_success_cn}/{cn}={100*transfer_success_cn/cn:.4f}%',
              file=f)


def resnet_test152a50(extract:bool=True):
    net_structure_src = 'Resnet152'
    net_structure_tgt_list = ['Resnet50']
    net_src = load_resnet152()
    if extract:
        get_src_net_trigger_feature(net_src, net_structure_src)
    net_tgt = load_resnet50()
    net_list = [net_tgt]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def resnet_test50a152(extract:bool=True):
    net_structure_src = 'Resnet50'
    net_structure_tgt_list = ['Resnet152']
    net_src = load_resnet50()
    if extract:
        get_src_net_trigger_feature(net_src, net_structure_src)
    net_tgt = load_resnet152()
    net_list = [net_tgt]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def vit_test32a16(extract:bool=True):
    net_structure_src = 'ViT-b-32'
    net_structure_tgt_list = ['ViT-b-16']
    net_src = load_vit_b_32()
    if extract:
        get_src_net_trigger_feature(net_src, net_structure_src)
    net_tgt = load_vit_b_16()
    net_list = [ net_tgt]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def vit_test16a32(extract:bool=True):
    net_structure_src = 'ViT-b-16'
    net_structure_tgt_list = ['ViT-b-32']
    net_src = load_vit_b_16()
    if extract:
        get_src_net_trigger_feature(net_src, net_structure_src)
    net_tgt = load_vit_b_32()
    net_list = [ net_tgt]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def res2vit_test_152a16():
    net_structure_src = 'Resnet152'
    net_structure_tgt_list = ['ViT-b-16']
    net_src = load_resnet152()
    net_tgt1 = load_vit_b_16()
    #net_tgt2 = load_vit_b_32()
    net_list = [net_tgt1]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def res2vit_test_152a32():
    net_structure_src = 'Resnet152'
    net_structure_tgt_list = ['ViT-b-32']
    net_src = load_resnet152()
    net_tgt1 = load_vit_b_32()
    #net_tgt2 = load_vit_b_32()
    net_list = [net_tgt1]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def res2vit_test_50a32():
    net_structure_src = 'Resnet50'
    net_structure_tgt_list = ['ViT-b-32']
    net_src = load_resnet50()
    net_tgt1 = load_vit_b_32()
    #net_tgt2 = load_vit_b_32()
    net_list = [net_tgt1]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def res2vit_test_50a16():
    net_structure_src = 'Resnet50'
    net_structure_tgt_list = ['ViT-b-16']
    net_src = load_resnet50()
    net_tgt1 = load_vit_b_16()
    #net_tgt2 = load_vit_b_32()
    net_list = [net_tgt1]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def vit2res_test_32a152():
    net_structure_src = 'ViT-b-32'
    net_structure_tgt_list = ['Resnet152']
    net_src = load_vit_b_32()
    net_tgt2 = load_resnet152()
    net_list = [net_tgt2]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def vit2res_test_32a50():
    net_structure_src = 'ViT-b-32'
    net_structure_tgt_list = ['Resnet50']
    net_src = load_vit_b_32()
    net_tgt2 = load_resnet50()
    net_list = [net_tgt2]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def vit2res_test_16a152():
    net_structure_src = 'ViT-b-16'
    net_structure_tgt_list = ['Resnet152']
    net_src = load_vit_b_16()
    net_tgt2 = load_resnet152()
    net_list = [net_tgt2]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


def vit2res_test_16a50():
    net_structure_src = 'ViT-b-16'
    net_structure_tgt_list = ['Resnet50']
    net_src = load_vit_b_16()
    net_tgt2 = load_resnet50()
    net_list = [net_tgt2]
    for i in range(len(net_list)):
        test_transfer(net_src, net_list[i], net_structure_src, net_structure_tgt_list[i])


if __name__ == "__main__":
    import sys
    """if len(sys.argv) < 2:
        print("No model specified")
        name = 'resnet'
    else:
        name = sys.argv[1]
        print(f"{name} is running")"""
    with torch.no_grad():
        for name in ['resnet','vit']:
            if name.lower() == 'resnet':
                resnet_test50a152(True)
                resnet_test152a50(True)
                res2vit_test_152a16()
                res2vit_test_152a32()
                res2vit_test_50a16()
                res2vit_test_50a32()
            elif name.lower() == 'vit':
                vit_test32a16(True)
                vit_test16a32(True)
                vit2res_test_16a152()
                vit2res_test_32a152()
                vit2res_test_16a50()
                vit2res_test_32a50()
                

