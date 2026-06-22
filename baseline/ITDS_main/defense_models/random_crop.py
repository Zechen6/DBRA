import torch
import torchvision.transforms.functional as TF

def random_crop_defense(images, crop_size=28):
    """
    images: [B,C,H,W]
    return: [B,C,H,W]
    """
    B, C, H, W = images.shape

    outputs = []

    for i in range(B):
        top = torch.randint(0, H - crop_size + 1, (1,)).item()
        left = torch.randint(0, W - crop_size + 1, (1,)).item()

        cropped = images[i:i+1, :, top:top+crop_size, left:left+crop_size]

        resized = torch.nn.functional.interpolate(
            cropped,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )

        outputs.append(resized)

    return torch.cat(outputs, dim=0)
