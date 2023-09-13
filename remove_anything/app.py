import os
import urllib.request

import gradio as gr
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .fcf import FCF
from .lama import LaMa
from .ldm import LDM
from .mat import MAT


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str) -> None:
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
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


def predict(inputs):
    image = inputs["image"].convert("RGB")
    mask = inputs["mask"].convert("L")

    image = np.array(image).astype('float32') / 255
    mask = np.array(mask).astype('float32')[:,:,None] / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    
    output = inpainter(image, mask)

    output = Image.fromarray(output)
    return output


def download_ckpt(model, ckpt):
    if ckpt is not None:
        return ckpt
    
    ckpt_path = INPAINTERS[model]['ckpt']
    name = INPAINTERS[model]['name']
    download_url(
        ckpt_path,
        os.path.join(
            os.environ.get('REMOVE_ANYTHING_CHECKPOINT_DIR', 'checkpoints'),
            name,
        ),
    )


def main(args):
    inputs = gr.Image(tool="sketch", label="Input", type="pil")
    outputs = gr.Image(type="pil", label="output")

    title = "Remove Anything"

    gr.Interface(
        predict,
        inputs,
        outputs,
        title=title,
    ).launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Remove Anything')
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint Path')
    parser.add_argument('--model', type=str, default='lama', choices=INPAINTERS.keys(), help='Model type')
    parser.add_argument('--port', type=int, default=10086, help='Port')
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='IP address')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    ckpt_path = download_ckpt(args.model, args.ckpt)
    inpainter = INPAINTERS[args.model]['model'](
        ckpt_path=ckpt_path, device=torch.device(args.device)
    )

    main(args)
