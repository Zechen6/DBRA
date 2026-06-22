import torch.nn.functional as F

def median_smoothing(images, kernel_size=3):

    padding = kernel_size // 2

    unfolded = F.unfold(
        images,
        kernel_size=kernel_size,
        padding=padding
    )

    unfolded = unfolded.view(
        images.size(0),
        images.size(1),
        kernel_size*kernel_size,
        images.size(2)*images.size(3)
    )

    median = unfolded.median(dim=2)[0]

    return median.view_as(images)