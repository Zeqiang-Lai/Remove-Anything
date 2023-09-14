import os
import dola
import torch

from .fcf import FCF
from .lama import LaMa
from .ldm import LDM
from .mat import MAT


def download_ckpt(model, ckpt):
    if ckpt is not None:
        return ckpt

    ckpt_path = INPAINTERS[model]['ckpt']
    name = INPAINTERS[model]['name']
    dola.download_url(
        ckpt_path,
        os.path.join(
            os.environ.get('REMOVE_ANYTHING_CHECKPOINT_DIR', 'checkpoints'),
            name,
        ),
    )


INPAINTERS = dict(
    ldm=dict(
        model=LDM,
        ckpt='https://huggingface.co/aaronb/remove_anything/resolve/main/big-lama.pt',
        name='big-lama.pt',
    ),
    mat=dict(
        model=MAT,
        ckpt='https://huggingface.co/aaronb/remove_anything/resolve/main/mat_places_512.pth',
        name='mat_places_512.pth',
    ),
    fcf=dict(
        model=FCF,
        ckpt='https://huggingface.co/aaronb/remove_anything/resolve/main/fcf_places_512.pth',
        name='fcf_places_512.pth',
    ),
    lama=dict(
        model=LaMa,
        ckpt='https://huggingface.co/aaronb/remove_anything/resolve/main/ldm_inpainting_big.ckpt',
        name='ldm_inpainting_big.ckpt',
    ),
)


available_models = INPAINTERS.keys()


def load(model, ckpt_path, device):
    ckpt_path = download_ckpt(model, ckpt_path)
    inpainter = INPAINTERS[model]['model'](
        ckpt_path=ckpt_path, device=torch.device(device)
    )
    return inpainter
