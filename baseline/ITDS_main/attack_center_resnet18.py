import sys
cwd = ''
sys.path.insert(0, cwd)

from baseline.ITDS_main.cus_logits import *
from baseline.ITDS_main.attacks.config import *
from baseline.ITDS_main.attacks.itds import ITDS
from baseline.ITDS_main.defense_models.feature_squeezing import bit_depth_reduction
from baseline.ITDS_main.defense_models.JPEG import jpeg_compress
from baseline.ITDS_main.defense_models.median_smoothing import median_smoothing
from baseline.ITDS_main.defense_models.random_crop import random_crop_defense
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from confs.device_conf import device
from utils.load_utils import load_svhn_sources, load_cifar10_sources, load_clean_cifar10
import time
import math
import copy
import numpy as np
cifar10_root = '../NonPoisonBackdoor/clean_dataset'

def random_indices(L, size, device='cpu'):
    return torch.randint(
        low=0,
        high=L,
        size=(size,),
        device=device
    )


def main():
    vic_model_path = 'resnet18_cifar10_seed114514.pth'
    adv_model_path = 'resnet18_cifar10_seed0.pth'

    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=cifar10_root,
        train=True,
        download=True,
        transform=transform_train,
    )

    testset = torchvision.datasets.CIFAR10(
        root=cifar10_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
    )
    vic_nets = []
    vic_model = resnet18(weights=None).cuda()
    vic_model.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    )

    vic_model.maxpool = nn.Identity()

    vic_model.fc = nn.Linear(512, 10)

    vic_model = vic_model.to(device)
    adv_model = copy.deepcopy(vic_model)
    adv_model.load_state_dict(torch.load(adv_model_path))
    vic_model.load_state_dict(torch.load(vic_model_path))
    vic_model.eval().cuda()
    adv_model.eval().cuda()
    vic_nets.append(vic_model)

    target_class = 0

    target_imgs = []
    target_labels = []
    with torch.no_grad():
        for imgs, labs in testloader:
            imgs = imgs.to(device)
            labs = labs.to(device)
            temp = vic_model(imgs).argmax(-1)
            mask = temp == target_class

            if mask.sum() > 0:
                target_imgs.append(imgs[mask])
                target_labels.append(labs[mask])
                #if len(target_imgs) >= 100:
                #    break

    target_imgs = torch.cat(target_imgs).cuda()
    target_labels = torch.cat(target_labels).cuda()


    print("Target pool size:", len(target_imgs))

    # ----------------------------------
    # evaluation
    # ----------------------------------
    

    eps = 0.93
    alpha = 0.82
    attacker = ITDS(
        model=adv_model,
        eps=eps,
        alpha=alpha,
        steps=math.floor(eps/alpha),
        TI=True,
        DI=True
    )

    attack_success_list = [0 for _ in range(len(vic_nets))]
    asr_jpeg_list = [0 for _ in range(len(vic_nets))]
    asr_rd_crop_list = [0 for _ in range(len(vic_nets))]
    asr_median_smooth_list = [0 for _ in range(len(vic_nets))]
    time_sum = 0
    cn = 0

    for images, labels in testloader:

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

    with open(f'cache/baseline_situation_center-ITDS.log','a') as f:

        print("Time:\t",end_time-start_time,file=f)
        print("Normal ASR:\t", np.array(attack_success_list)/cn,file=f)
        print("JPEG ASR:\t", np.array(asr_jpeg_list)/cn,file=f)
        print("Smooth ASR:\t", np.array(asr_median_smooth_list)/cn,file=f)
        print("Rand Crop ASR:\t", np.array(asr_rd_crop_list)/cn,file=f)


if __name__ == "__main__":
    main()