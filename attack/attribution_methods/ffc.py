"""
This module implements FFC (Frequency Feature Contribution) utilities.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from confs.device_conf import device
import torch
import torch.nn as nn
import math


def select_bottom_element(scores:torch.Tensor,
                          ratio:float):
    """Select low-scoring features according to `ratio`."""
    assert ratio <= 1
    scores4sort = scores.view(scores.shape[0],-1)
    thred_idx = torch.tensor(math.ceil(scores4sort.shape[-1]*ratio)).to(scores.device).unsqueeze(-1)
    values,_ = torch.sort(scores4sort, dim=-1,descending=False)
    thred_values = values[:,thred_idx]
    mask = torch.where(scores4sort>thred_values,1,0)
    mask = mask.view(scores.shape)
    return mask


def select_top_element(scores:torch.Tensor, ratio:float):
    """Select top-scoring features according to `ratio`."""
    assert ratio <= 1
    scores4sort = scores.view(scores.shape[0],-1)
    thred_idx = torch.tensor(math.ceil(scores4sort.shape[-1]*ratio)).to(scores.device).unsqueeze(-1)
    values,_ = torch.sort(scores4sort, dim=-1,descending=True)
    thred_values = values[:,thred_idx]
    mask = torch.where(scores4sort>thred_values,1,0)
    mask = mask.view(scores.shape)
    return mask
    

def malicious_ffc(net:nn.Module, sample:torch.Tensor, 
                  tgt_label:torch.Tensor, lr=1000, echo=100):
    """Find features in the original sample that are most similar to the target class via adversarial optimization."""
    y = tgt_label*torch.ones(sample.shape[0]).to(device).long()
    with torch.enable_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        data_new = sample.clone()
        data_new.requires_grad = True
        for e in range(echo):
            pred = net(data_new)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            grad = data_new.grad.data.clone()        
            with torch.no_grad():
                data_new -= lr*grad
                data_new.grad.zero_()
        ori_freq = torch.fft.fft2(sample)
        new_freq = torch.fft.fft2(data_new)
        mag_ori = torch.abs(ori_freq)
        ori_after_mutual_energy = 2*(torch.conj(new_freq)*ori_freq).real       

        scores = (ori_after_mutual_energy/(mag_ori+1e-7)-mag_ori)
    return scores



def ffc(net:nn.Module, sample:torch.Tensor, lr=1000, echo=100):
    """For a single sample, find the frequency components with highest attribution using FFC."""
    net.eval()
    # torch.use_deterministic_algorithms(True) -- do not enable when using Smooth defenses

    with torch.no_grad():
        pred_label = net(sample).argmax(-1)
    with torch.enable_grad():
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1)
        data_new = sample.clone()
        data_new.requires_grad = True
        score_last = torch.fft.fft2(sample).abs()
        neg_score_sum = 0
        for _ in range(echo):
            pred = net(data_new)
            loss = loss_fn(pred, pred_label)
            optimizer.zero_grad()
            loss.backward()
            grad = data_new.grad.data.clone()
            with torch.no_grad():
                data_new -= lr*grad
                data_new.grad.zero_()
        ori_freq = torch.fft.fft2(sample)
        new_freq = torch.fft.fft2(data_new)
        mag_ori = torch.abs(ori_freq)
        ori_after_mutual_energy = 2*(torch.conj(new_freq)*ori_freq).real       
        scores = (ori_after_mutual_energy/(mag_ori+1e-7)-mag_ori)
    #scores += neg_score_sum
    return scores


def find_most_important_fea_by_mag(samples:torch.Tensor, 
                                net:nn.Module,
                                _):
    with torch.no_grad():
        net.eval()
        original_pred = net(samples).argmax(-1)
        scores = torch.fft.fft2(samples).abs()
        founded_flag = torch.zeros(samples.shape[0]).to(device)
        ratio4samples = torch.zeros(samples.shape[0]).to(device)
        step = 0.01
        filtered_sample_list = samples.clone()
        for e in range(30):
            masks = select_top_element(scores, (e+1)*step)
            freq_sample = torch.fft.fft2(samples)
            masked_sample = freq_sample*masks
            filtered_sample = torch.fft.ifft2(masked_sample).real
            new_pred = net(filtered_sample).argmax(-1)
            maintained_flag = (new_pred==original_pred).int()
            new_maintained = maintained_flag-founded_flag
            if (new_maintained==1).sum() > 0:
                filtered_sample_list[new_maintained==1] = filtered_sample[new_maintained==1]
            ratio4samples[new_maintained==1] = (e+1)*step
            founded_flag = torch.where(maintained_flag>founded_flag,maintained_flag, founded_flag)

    return scores, filtered_sample_list


def find_most_important_feature(samples:torch.Tensor, 
                                net:nn.Module,
                                ffc_params:dict={'lr':1000, 'echo':10}):
    """Select the minimal high-scoring frequency features that preserve the network's original prediction using FFC."""
    with torch.no_grad():
        net.eval()
        original_pred = net(samples).argmax(-1)
        scores = ffc(net, samples, lr=ffc_params['lr'], echo=ffc_params['echo'])
        founded_flag = torch.zeros(samples.shape[0]).to(device)
        ratio4samples = torch.zeros(samples.shape[0]).to(device)
        step = 0.01
        filtered_sample_list = samples.clone()
        for e in range(30):
            masks = select_top_element(scores, (e+1)*step)
            freq_sample = torch.fft.fft2(samples)
            masked_sample = freq_sample*masks
            filtered_sample = torch.fft.ifft2(masked_sample).real
            new_pred = net(filtered_sample).argmax(-1)
            maintained_flag = (new_pred==original_pred).int()
            new_maintained = maintained_flag-founded_flag
            if (new_maintained==1).sum() > 0:
                filtered_sample_list[new_maintained==1] = filtered_sample[new_maintained==1]
            ratio4samples[new_maintained==1] = (e+1)*step
            founded_flag = torch.where(maintained_flag>founded_flag,maintained_flag, founded_flag)

    return scores, filtered_sample_list


def find_top_malicious_feature(samples:torch.Tensor, 
                                net:nn.Module,
                                tgt_label:torch.Tensor|int,
                                ffc_param:dict={'lr':1000, 'echo':10}):
    """Use malicious FFC to select minimal high-scoring frequency features that steer prediction to `tgt_label`."""
    filtered_sample_tensor = torch.zeros_like(samples)
    with torch.no_grad():
        net.eval()
        scores = malicious_ffc(net, samples, lr=ffc_param['lr'], echo=ffc_param['echo'], tgt_label=tgt_label)

        step = 0.02

        for e in range(10):
            masks = select_top_element(scores, (e+1)*step)
            freq_sample = torch.fft.fft2(samples)
            masked_sample = freq_sample*masks
            filtered_sample = torch.fft.ifft2(masked_sample).real
            new_pred = net(filtered_sample).argmax(-1)
            maintained_flag = (new_pred==tgt_label)
            if maintained_flag.sum() > 0:
                filtered_sample_tensor[maintained_flag] = filtered_sample[maintained_flag]

    return scores, filtered_sample_tensor


def recurrent_filt(net:nn.Module, 
                   sample:torch.Tensor,
                   iters:int):
    """Refine feature selection by repeatedly applying FFC to filter key features."""
    scores, features = find_most_important_feature(sample,net)
    for _ in range(iters):
        scores, features = find_most_important_feature(features, net)
    return scores, features


def malicious_recurrent_filt(net:nn.Module, 
                            sample:torch.Tensor,
                            iters:int,
                            tgt_label:int):
    """Refine malicious feature selection by repeatedly applying malicious FFC toward `tgt_label`."""
    scores, features = find_top_malicious_feature(sample,net,tgt_label)
    for _ in range(iters):
        scores, features = find_top_malicious_feature(features, net, tgt_label)
    return scores, features
