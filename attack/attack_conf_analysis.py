import sys
cwd = '。。/'
sys.path.insert(0, cwd)

from confs.device_conf import device
from torchvision.models import resnet18, resnet50
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np
import torch.nn as nn
from attack.attribution_methods.ffc import find_most_important_feature, recurrent_filt,malicious_recurrent_filt
from attack.attribution_methods.ffc import find_top_malicious_feature
from confs.implantation_confs import *
from attack.decision_basis_replacement import refined_trigger_implantation as refined_trigger_implantation_analysis
from utils.load_utils import load_svhn_sources, load_cifar10_sources


LOG_NAME = ''

CIFAR_CLASSNUM = 10


def construct_dict_by_querying(
        samples:torch.Tensor,
        model:nn.Module,
):

    model.eval()
    with torch.no_grad():
        outputs = model(samples)
        preds = outputs.argmax(dim=-1)
    
    potential_triggers = preds == target_label
    

    return potential_triggers


def attack_main(fed_name:str, 
                model_name:str, 
                dataset_name:str,
                victim_id=victim_client):
    """Main function orchestrating the attack; calls helper routines to perform steps."""
    # Load data and models
    if dataset_name.lower() == 'cifar10':
        train_loader, test_loader, \
        client_train_loader, client_test_loader, \
        adv_model, victim_model = load_cifar10_sources(fed_name, model_name,victim_id)

    elif dataset_name.lower() == 'svhn':
        train_loader, test_loader, \
        client_train_loader, client_test_loader, \
        adv_model, victim_model = load_svhn_sources(fed_name, model_name,victim_id)
    adv_model.eval()
    victim_model.eval()
    trigger_dict_data = []
    for b, (X,y) in enumerate(client_train_loader):
        trigger_dict_data.append(X.to(device))
    

    trigger_dict_data = torch.cat(trigger_dict_data, dim=0)
    pred = adv_model(trigger_dict_data).argmax(-1)
    # Remove samples that the model itself fails to classify correctly

    trigger_dict_data = trigger_dict_data[pred == target_label]

    local_success_flag \
        = construct_dict_by_querying(trigger_dict_data, adv_model)
    local_success_samples = trigger_dict_data[local_success_flag]
    
    print(f"Potential triggers found in dictionary construction:\
           {local_success_flag.sum().item()}/{len(pred)}")

    both_success_samples = local_success_samples

    cn = 0
    tgt_label = torch.tensor(target_label).to(device)
    adv_model.eval()
    victim_model.eval()
    max_len = 30
    interval = 3
    suc_interval_list = [0 for i in range(max_len // interval)]
    suc_accumulate_list = [0 for i in range(max_len // interval)]
    # This version includes early trigger feature extraction
    iter_num = 2
    for _ in range(iter_num):
        malicious_score, both_success_samples \
            = find_most_important_feature(both_success_samples, adv_model, 
                                        ffc_params=trigger_extraction_ffc_params)

    for batch_idx, (X, y) in enumerate(test_loader):
        data, label = X.to(device), y.to(device)
        data = data[label!=tgt_label]
        label = label[label!=tgt_label]
        if len(data) == 0:
            continue

        for i in range(data.shape[0]):
            cn += 1
            suc_accumulate_list_tmp, suc_interval_list_tmp, \
                _, _, _\
                 = \
                refined_trigger_implantation_analysis(data[i].unsqueeze(0), adv_model, tgt_label, 
                                            both_success_samples, victim_model,
                                            malicious_score)
            suc_accumulate_list = [x+y for x,y in zip(suc_accumulate_list, suc_accumulate_list_tmp)]
            suc_interval_list = [x+y for x,y in zip(suc_interval_list, suc_interval_list_tmp)]
            print(f"Current ASR:{suc_accumulate_list}, \
                  Current Interval Suc Rate:{suc_interval_list}")
            
    return suc_accumulate_list, suc_interval_list

     
if __name__ == "__main__":
    # FedBN, FedALA, FedRep, FedCAC, SCAFFOLD, Ditto
    import sys
    if len(sys.argv) < 2:
        print("No federated learning algorithm specified")
        name = 'FedCAC'
    else:
        name = sys.argv[1]
        print(f"{name} is runnning")
    
    with torch.no_grad():
        attack_res = {}
        if ATTACK_ALL:
            for vic_id in victim_client:
                attack_res[vic_id] = attack_main(name, 'resnet18', 'Cifar10', vic_id)
                print(attack_res)
        else:
            attack_main(name, 'resnet18', 'Cifar10')
    with open('attack_conf.log'
              ,'a') as f:
        print(attack_res, file=f)
