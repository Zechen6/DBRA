import sys
cwd = ''
sys.path.insert(0, cwd)

import warnings
from baseline.PUAP.universal_pert import universal_perturbation
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from utils.load_utils import load_svhn_sources, load_cifar10_sources
import time
from datetime import datetime, timezone

from baseline.ITDS_main.defense_models.JPEG import jpeg_compress
from baseline.ITDS_main.defense_models.median_smoothing import median_smoothing
from baseline.ITDS_main.defense_models.random_crop import random_crop_defense


device = 'cuda:0'
epsilon = 254.0/ 255.0

mean = [0.4914, 0.4822, 0.4465]       # CIFAR-10 mean
std  = [0.2023, 0.1994, 0.2010]       # CIFAR-10 std
# std = [1.0, 1.0, 1.0]
adv_id = 0
victim_list = list(range(20))
victim_list.remove(adv_id)
vic_net_list = []
target_label = 0
def construct_dict_by_querying(
        samples:torch.Tensor,
        tgt_model:nn.Module,
):

    tgt_model.eval()
    with torch.no_grad():
        outputs = tgt_model(samples)
        preds = outputs.argmax(dim=-1)
    
    potential_triggers = preds == target_label
    
    return potential_triggers


for vid in victim_list:
    train_loader, test_loader, \
    client_train_loader, client_test_loader, \
    adv_model, victim_model = load_cifar10_sources('Ditto', 'resnet18',vid)
    adv_model.eval()
    victim_model.eval()
    
    vic_net_list.append(victim_model)

trigger_dict_data = []
for b, (X,y) in enumerate(client_train_loader):
    trigger_dict_data.append(X.to(device))


trigger_dict_data = torch.cat(trigger_dict_data, dim=0)
pred = adv_model(trigger_dict_data).argmax(-1)

trigger_dict_data = trigger_dict_data[pred == target_label]
local_success_flag \
    = construct_dict_by_querying(trigger_dict_data, adv_model)

local_success_samples = trigger_dict_data[local_success_flag]
attack_success_list = [0 for _ in range(len(vic_net_list))]
asr_jpeg_list = [0 for _ in range(len(vic_net_list))]
asr_rd_crop_list = [0 for _ in range(len(vic_net_list))]
asr_median_smooth_list = [0 for _ in range(len(vic_net_list))]

print("Constructing Perturbation")
utc_now = datetime.now(timezone.utc)
print(utc_now)  # 
start_time = time.time()
v = universal_perturbation(local_success_samples, test_loader, adv_model, xi = epsilon)
end_time = time.time()
print(f"Use {end_time-start_time}s to construct")
attack_success_list = [0 for _ in range(len(vic_net_list))]
with torch.no_grad():
    for batch_idx, (X,y) in enumerate(test_loader):
        X = X[y!=target_label]
        X = X.to(device)+torch.tensor(v).cuda()
        jpeg_img = jpeg_compress(X)
        #med_smo_img = median_smoothing(X)
        rd_crop_img = random_crop_defense(X)
        for i in range(len(vic_net_list)):
            pred = vic_net_list[i](X).argmax(-1)
            attack_success_list[i] += (pred==target_label).sum().item()
            pred = vic_net_list[i](jpeg_img).argmax(-1)
            asr_jpeg_list[i] += (pred==target_label).sum().item()
            pred = vic_net_list[i](rd_crop_img).argmax(-1)
            asr_rd_crop_list[i] += (pred==target_label).sum().item()

    
asr_array = np.array(attack_success_list)
print(asr_array/9000)
asr_array = np.array(asr_jpeg_list)
print(asr_array/9000)
asr_array = np.array(asr_rd_crop_list)
print(asr_array/9000)







