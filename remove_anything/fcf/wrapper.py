import os
import sys

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from models.networks import Generator


def make_batch(image, mask, device):
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(device)

    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask = torch.from_numpy(mask).to(device)
    
    return image, mask


class FCF:
    def __init__(self, ckpt_path, device, resolution=512, truncation_psi=1):
        print('Loading networks from "%s"...' % ckpt_path)
        
        encoder_kwargs = {'block_kwargs': {}, 'mapping_kwargs': {}, 'epilogue_kwargs': {'mbstd_group_size': 4}, 'channel_base': 32768, 'channel_max': 512, 'num_fp16_res': 4, 'conv_clamp': 256}
        mapping_kwargs = {'num_layers': 2}
        synthesis_kwargs = {'channel_base': 32768, 'channel_max': 512, 'num_fp16_res': 4, 'conv_clamp': 256}
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3,
                           encoder_kwargs=encoder_kwargs, mapping_kwargs=mapping_kwargs, synthesis_kwargs=synthesis_kwargs).to(device).eval()
        self.G.load_state_dict(torch.load(ckpt_path))
        self.truncation_psi = truncation_psi
        self.resolution = resolution
        self.device = device
        
    @torch.no_grad()
    def __call__(self, image, mask):
        device = self.device
    
        image, mask = make_batch(image, mask, device)
         
        label = torch.zeros([1, self.G.c_dim], device=device)

        erased_img = image.clone()
        erased_img = erased_img * (1 - mask)

        pred_img = self.G(img=torch.cat([0.5 - mask, erased_img], dim=1), c=label, 
                                  truncation_psi=self.truncation_psi, noise_mode='const')
        output = mask.to(device) * pred_img + (1 - mask).to(device) * image.to(device)
            
        output = (output.permute(0, 2, 3, 1) * 127.5 + 127.5).round().clamp(0, 255).to(torch.uint8)
        output = output[0].cpu().numpy()
        
        return output