import sys
cwd = '/data01/lzc/Experiments/PFLExtractionAttack/'
sys.path.insert(0, cwd)

from baseline.ITDS_main.cus_logits import *
from baseline.ITDS_main.attacks.config import *
from baseline.ITDS_main.attacks.itds import ITDS
from confs.device_conf import device
from utils.load_utils import load_svhn_sources, load_cifar10_sources, load_clean_cifar10
import time
import math

def random_indices(L, size, device='cpu'):
    return torch.randint(
        low=0,
        high=L,
        size=(size,),
        device=device
    )


def main():
    victim_id = 3
    fed_name='Ditto'
    model_name = 'resnet18'
    train_loader, test_loader, \
    client_train_loader, client_test_loader, \
    model, victim_model = load_cifar10_sources(fed_name, model_name,victim_id)
    model.eval().cuda()
    images, labels = next(iter(test_loader))

    images = images.cuda()
    labels = labels.cuda()

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
    _,test_loader = load_clean_cifar10(batch_size=360, shuffle=True)

    print("Target pool size:", len(target_imgs))

    # ----------------------------------
    # evaluation
    # ----------------------------------
    
    for epsilon in range(2,100):
        for alpha in range(2,100):
            if alpha > epsilon:
                continue
            attacker = ITDS(
                model=model,
                eps=epsilon/100,
                alpha=alpha/100,
                steps=math.floor(epsilon/alpha),
                TI=True,
                DI=True
            )

            attack_success = 0
            attack_total = 0
            time_sum = 0
            cn = 0
            thred = 100
            _,test_loader = load_clean_cifar10(batch_size=128, shuffle=True)
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
                    with torch.no_grad():
                        pred_adv = victim_model(attacker.trf(adv_images)).argmax(1)

                    attack_success += 1 if (pred_adv == target_class).sum().item() > 0 else 0
                    attack_total += 1
                    print("Time:\t",end_time-start_time)
                    print("ASR:\t",attack_success/attack_total)
                    cn += 1
                    if cn > thred:
                        break
            asr = 100 * attack_success / attack_total

            print("=" * 60)
            print(f"Target ASR: {asr:.2f}%")
            print("=" * 60)
            print(time_sum)
            with open('cache/baseline_situation.log','a') as f:
                print("Epsilon:",epsilon/100,file=f)
                print("Alpha:",alpha/100,file=f)
                print("=" * 60, file=f)
                print(f"Target ASR: {asr:.2f}%", file=f)
                print("=" * 60, file=f)
                print("Total Time:",time_sum, file=f)


if __name__ == "__main__":
    main()