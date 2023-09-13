import numpy as np
import torch

from .networks.mat import Generator


def make_batch(image, mask, device):
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)

    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask = 1-mask
    mask = torch.from_numpy(mask).to(device)
    
    return image, mask


class MAT:
    def __init__(self, ckpt_path, device, resolution=512, truncation_psi=1):
        print(f'Loading networks from: {ckpt_path}')
        net_res = 512 if resolution > 512 else resolution
        self.inpainter = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=net_res, img_channels=3).to(device).eval()
        self.inpainter.load_state_dict(torch.load(ckpt_path))
        self.resolution = resolution
        self.device = device
        self.truncation_psi = truncation_psi
        
        
    @torch.no_grad()
    def __call__(self, image, mask):
        device = self.device
        
        noise_mode = 'const'
        if self.resolution != 512:
            noise_mode = 'random'
    
        image, mask = make_batch(image, mask, device)

        z = torch.from_numpy(np.random.randn(1, self.inpainter.z_dim)).to(device)
        label = torch.zeros([1, self.inpainter.c_dim], device=device)
        output = self.inpainter(image, mask, z, label, truncation_psi=self.truncation_psi, noise_mode=noise_mode)
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()

        return output