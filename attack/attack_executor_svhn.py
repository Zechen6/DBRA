import sys
cwd = '。。/'
sys.path.insert(0, cwd)

from confs.device_conf import device
from torchvision.models import resnet18, resnet50
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from attack.attribution_methods.ffc import find_most_important_feature, recurrent_filt,malicious_recurrent_filt
from attack.attribution_methods.ffc import find_top_malicious_feature
from confs.data_conf import cifar10_root
from confs.implantation_confs import *
from attack.DBRA_main import refined_trigger_implantation as refined_trigger_implantation_analysis
from utils.load_utils import load_svhn_sources, load_cifar10_sources

import copy

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
    asr_cn = 0
    cn = 0
    tgt_label = torch.tensor(target_label).to(device)
    adv_model.eval()
    victim_model.eval()
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
            _,transfer_success, local_suc_flag = \
                refined_trigger_implantation_analysis(data[i].unsqueeze(0), adv_model, tgt_label, 
                                            both_success_samples, victim_model,
                                            malicious_score)
            asr_cn += 1 if transfer_success > 0 else 0
            print(f'Sample {cn} processed.')
            print(f'Success triggers found: {asr_cn}/{cn}={100*asr_cn/cn:.4f}%')

    return 100*asr_cn/cn

     
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
                attack_res[vic_id]=attack_main(name, 'resnet50', 'SVHN', vic_id)
                print(attack_res)
        else:
            attack_main(name, 'resnet50', 'SVHN')
    with open('final_asr.log'
              ,'a') as f:
        print(attack_res, file=f)
