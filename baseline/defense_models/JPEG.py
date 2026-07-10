import io
from PIL import Image
import torch
import torchvision.transforms as T

def jpeg_compress(images, quality=75):

    device = images.device

    outputs = []

    for img in images:

        img_pil = T.ToPILImage()(img.cpu())

        buffer = io.BytesIO()

        img_pil.save(
            buffer,
            format='JPEG',
            quality=quality
        )

        buffer.seek(0)

        img_jpeg = Image.open(buffer)

        img_tensor = T.ToTensor()(img_jpeg)

        outputs.append(img_tensor)

    return torch.stack(outputs).to(device)