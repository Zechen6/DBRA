import torch

def bit_depth_reduction(images, bits=1):

    levels = 2 ** bits - 1

    images = torch.round(images * levels) / levels

    return images.clamp(0,1)