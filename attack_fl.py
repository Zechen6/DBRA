import sys
cwd = '/data01/lzc/Experiments/PFLExtractionAttack/'
sys.path.insert(0, cwd)

from baseline.ITDS_main.cus_logits import *
from baseline.ITDS_main.attacks.config import *
from baseline.ITDS_main.attacks.itds import ITDS
from baseline.ITDS_main.defense_models.feature_squeezing import bit_depth_reduction
from baseline.ITDS_main.defense_models.JPEG import jpeg_compress
from baseline.ITDS_main.defense_models.median_smoothing import median_smoothing
from baseline.ITDS_main.defense_models.random_crop import random_crop_defense
from confs.device_conf import device
from utils.load_utils import load_svhn_sources, load_cifar10_sources, load_clean_cifar10
import time
import math
import numpy as np

def random_indices(L, size, device='cpu'):
    return torch.randint(
        low=0,
        high=L,
        size=(size,),
        device=device
    )


def main():
    
    fed_name='FedALA'
    model_name = 'resnet18'
    victim_id_list = list(range(20))
    victim_id_list.remove(0) # 移走攻击者
    vic_nets = []
    for vid in victim_id_list:
        train_loader, test_loader, \
        client_train_loader, client_test_loader, \
        model, victim_model = load_cifar10_sources(fed_name, model_name,vid)
        model.eval().cuda()
        victim_model.eval().cuda()
        vic_nets.append(victim_model)


    target_class = 0

    target_imgs = []
    target_labels = []
    with torch.no_grad():
        for imgs, labs in client_test_loader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            temp = model(imgs).argmax(-1)
            mask = temp == target_class

            if mask.sum() > 0:
                target_imgs.append(imgs[mask])
                target_labels.append(labs[mask])

    target_imgs = torch.cat(target_imgs).cuda()
    target_labels = torch.cat(target_labels).cuda()


    print("Target pool size:", len(target_imgs))

    # ----------------------------------
    # evaluation
    # ----------------------------------
    


    attacker = ITDS(
        model=model,
        eps=0.93,
        alpha=0.83,
        steps=math.floor(0.93/0.83),
        TI=True,
        DI=True
    )

    attack_success_list = [0 for _ in range(19)]
    asr_jpeg_list = [0 for _ in range(19)]
    asr_rd_crop_list = [0 for _ in range(19)]
    asr_median_smooth_list = [0 for _ in range(19)]
    time_sum = 0
    cn = 0

    for images, labels in test_loader:

        images = images.cuda()
        labels = labels.cuda()

        # clean accuracy

        # attack only non-target samples
        mask = labels != target_class

        if mask.sum() == 0:
            continue
        
        atk_images = images[mask]
        atk_labels = labels[mask]
        #idx = torch.randperm(len(target_imgs), device=device)[:len(atk_images)]
        for i in range(len(atk_images)):
            cn += 1
            atk_temp = atk_images[i].unsqueeze(0).repeat(target_imgs.shape[0],1,1,1)
            start_time = time.time()
            adv_images = attacker(
                atk_temp,
                atk_labels,
                target_imgs,
                target_labels
            )
            end_time = time.time()
            time_sum += end_time-start_time
            img_tensor = attacker.trf(adv_images)
            jpeg_img = jpeg_compress(img_tensor)
            med_smo_img = median_smoothing(img_tensor)
            rd_crop_img = random_crop_defense(img_tensor)
            with torch.no_grad():
                
                for j in range(len(vic_nets)):
                    pred_adv = vic_nets[j](img_tensor).argmax(1)
                    attack_success_list[j] += 1 if (pred_adv == target_class).sum().item() > 0 else 0
                for j in range(len(vic_nets)):
                    pred_adv = vic_nets[j](jpeg_img).argmax(1)
                    asr_jpeg_list[j] += 1 if (pred_adv == target_class).sum().item() > 0 else 0
                for j in range(len(vic_nets)):
                    pred_adv = vic_nets[j](med_smo_img).argmax(1)
                    asr_median_smooth_list[j] += 1 if (pred_adv == target_class).sum().item() > 0 else 0
                for j in range(len(vic_nets)):
                    pred_adv = vic_nets[j](rd_crop_img).argmax(1)
                    asr_rd_crop_list[j] += 1 if (pred_adv == target_class).sum().item() > 0 else 0
            print("Time:\t",end_time-start_time)
            print("Normal ASR:\t", np.array(attack_success_list)/cn)
            print("JPEG ASR:\t", np.array(asr_jpeg_list)/cn)
            print("Smooth ASR:\t", np.array(asr_median_smooth_list)/cn)
            print("Rand Crop ASR:\t", np.array(asr_rd_crop_list)/cn)

    with open(f'cache/baseline_situation{fed_name}.log','a') as f:

        print("Time:\t",end_time-start_time,file=f)
        print("Normal ASR:\t", np.array(attack_success_list)/cn,file=f)
        print("JPEG ASR:\t", np.array(asr_jpeg_list)/cn,file=f)
        print("Smooth ASR:\t", np.array(asr_median_smooth_list)/cn,file=f)
        print("Rand Crop ASR:\t", np.array(asr_rd_crop_list)/cn,file=f)


if __name__ == "__main__":
    main()