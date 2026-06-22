import sys
cwd = '../'
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
from net_structures.load_pfl_lib_models import load_fedrep_cifar10_models
from utils.pfl_dataset_utils import load_train_data, load_test_data
from confs.implantation_confs import *
from attack.DBRA_main import refined_trigger_implantation as refined_trigger_implantation_analysis
from attack.attribution_methods.ffc import select_top_element, select_bottom_element
from attack.trigger_extraction import load_clean_cifar10
from utils.load_utils import load_svhn_sources, load_cifar10_sources


LOG_NAME = ''

CIFAR_CLASSNUM = 10
FINTUNED_MODEL_DIR = ''


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



def attack_main(fed_name:str, model_name:str, dataset_name:str):
    """
    Main function orchestrating the attack; it calls helper routines to complete the attack flow.
    """
    # Load data and models
    if dataset_name.lower() == 'cifar10':
        train_loader, test_loader, \
        client_train_loader, client_test_loader, \
        adv_model, victim_model = load_cifar10_sources(fed_name, model_name,)
        finetuned_model_params = torch.load(
            f'{FINTUNED_MODEL_DIR}{fed_name}_cifar10_client{victim_client}_'\
            + 'repaired.pth'
        )

    elif dataset_name.lower() == 'svhn':
        train_loader, test_loader, \
        client_train_loader, client_test_loader, \
        adv_model, victim_model = load_svhn_sources(fed_name, model_name,)
        finetuned_model_params = torch.load(
            f'{FINTUNED_MODEL_DIR}{fed_name}_svhn_client{victim_client}_'\
            + 'repaired.pth'
        )
    victim_model.load_state_dict(finetuned_model_params)
    adv_model.eval()
    victim_model.eval()
    trigger_dict_data = []

    
    for b, (X,y) in enumerate(client_test_loader):
        trigger_dict_data.append(X.to(device))
    trigger_dict_data = torch.cat(trigger_dict_data, dim=0)
    pred = adv_model(trigger_dict_data).argmax(-1)
    # Remove samples that the model itself fails to classify correctly
    trigger_dict_data = trigger_dict_data[pred == target_label]

    local_success_flag \
        = construct_dict_by_querying(trigger_dict_data, adv_model)
    both_success_flag = local_success_flag

    both_success_samples = trigger_dict_data[both_success_flag]


    fail_samples = None
    asr_cn = 0
    cn = 0
    tgt_label = torch.tensor(target_label).to(device)
    adv_model.eval()
    victim_model.eval()
    suc_times = torch.zeros(both_success_samples.shape[0]).to(device)
    # This version includes early trigger feature extraction
    _, both_success_samples \
        = find_most_important_feature(both_success_samples, adv_model, 
                                      ffc_params=trigger_extraction_ffc_params)
    for batch_idx, (X, y) in enumerate(test_loader):
        data, label = X.to(device), y.to(device)
        data = data[label!=tgt_label]
        if len(data) == 0:
            continue

        for i in range(data.shape[0]):
            cn += 1
            _, transfer_success_flag, local_suc_flag = \
                refined_trigger_implantation_analysis(data[i].unsqueeze(0), adv_model, tgt_label, 
                                            both_success_samples, victim_model,
                                            suc_times,
                                            fail_samples)
            suc_times = torch.zeros(both_success_samples.shape[0]).to(device)
            asr_cn += 1 if transfer_success_flag.sum().item() > 0 else 0

            print(f'Sample {cn} processed.')
            print(f'Success Rate: {asr_cn}/{cn}={100*asr_cn/cn:.4f}%')

     
if __name__ == "__main__":
    with torch.no_grad():
        attack_main('FedCAC', 'resnet18', 'Cifar10')

