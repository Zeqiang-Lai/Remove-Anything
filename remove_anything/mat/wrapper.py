import numpy as np
import torch
import dola

from .networks.mat import Generator


def make_batch(image, mask, device):
    image = image.astype(np.float32) * 2 - 1
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)

    mask = mask.astype(np.float32)
    mask = mask[None].transpose(0, 3, 1, 2)

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
    def forward(self, image, mask):
        device = self.device
        
        noise_mode = 'const'
        if self.resolution != 512:
            noise_mode = 'random'
    
        image, mask = make_batch(image, mask, device)

        z = torch.from_numpy(np.random.randn(1, self.inpainter.z_dim)).to(device)
        label = torch.zeros([1, self.inpainter.c_dim], device=device)
        output = self.inpainter(image, mask, z, label, truncation_psi=self.truncation_psi, noise_mode=noise_mode)
        
        predicted_image = torch.clamp((output + 1.0) / 2.0, min=0.0, max=1.0)
        inpainted = predicted_image.cpu().numpy().transpose(0, 2, 3, 1)[0]

        return inpainted
    
    def __call__(self, image, mask):
        origin_height, origin_width = image.shape[:2]
        pad_image = dola.imresize(image, (512,512), mode='cubic')
        pad_mask = dola.imresize(mask, (512,512), mode='nearest')
        
        result = self.forward(pad_image, pad_mask)
        result = dola.imresize(result, (origin_height, origin_width), mode='cubic')

        # result = result * (mask) + image * (1 - mask)
        result = np.clip(result * 255, 0, 255).astype("uint8")
        return result
