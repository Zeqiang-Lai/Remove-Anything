import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
import dola

from .ldm.models.diffusion.ddim import DDIMSampler
from .ldm.util import instantiate_from_config

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)



def make_batch(image, mask, device):
    image = image.astype(np.float32)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = mask.astype(np.float32)
    mask = mask[None].transpose(0, 3, 1, 2)
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


class LDM:
    pad_mod = 32
    pad_to_square = False
    resize = 512

    def __init__(self, ckpt_path, device=torch.device("cuda"), ddim_steps=50):
        config = OmegaConf.load(os.path.join(CURRENT_DIR, "config.yaml"))
        model = instantiate_from_config(config.model)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)
        self.model = model.to(device)
        self.sampler = DDIMSampler(model)
        self.device = device
        self.ddim_steps = ddim_steps

    @torch.no_grad()
    def forward(self, image, mask):
        device = self.device
        model = self.model
        sampler = self.sampler

        with self.model.ema_scope():
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask
            c = model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(
                batch["mask"], size=c.shape[-2:]
            )
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1] - 1,) + c.shape[2:]
            samples_ddim, _ = sampler.sample(
                S=self.ddim_steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False,
            )
            x_samples_ddim = model.decode_first_stage(samples_ddim)

            predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            inpainted = predicted_image.cpu().numpy().transpose(0, 2, 3, 1)[0]

        return inpainted

    def __call__(self, image, mask):
        origin_height, origin_width = image.shape[:2]
        pad_image = dola.imresize(image, (512,512), mode='cubic')
        pad_mask = dola.imresize(mask, (512,512), mode='nearest')
        
        result = self.forward(pad_image, pad_mask)
        result = dola.imresize(result, (origin_height,origin_width), mode='cubic')

        result = result * mask + image * (1 - mask)
        result = np.clip(result * 255, 0, 255).astype("uint8")
        return result
