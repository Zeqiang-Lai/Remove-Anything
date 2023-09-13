# Remove Anything

A python package that includes some SOTA inpainting models that does not require text. 

- **Easy to use**: simple and unified API.
- **Self contained**: each method is self contained in a file or directory, easy to copy to use.

<a href="https://huggingface.co/aaronb/remove_anything">
    <img alt="huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Models-blue">
</a>

</br>

| Original                  | Mask                 | LaMa                 | LDM                  |
| ------------------------- | -------------------- | -------------------- | -------------------- |
| ![](assets/original.jpeg) | ![](assets/mask.png) | ![](assets/lama.png) | ![](assets/lama.png) |
| MAT                       | FCT                  |                      |                      |
| ![](assets/original.jpeg) | ![](assets/mask.png) |                      |                      |

## Install

```bash
pip install remove_anything
```

## Usage

The checkpoints would be automatically downloaded if using gradio app. For API usage, you could download checkpoint at [Huggingface](https://huggingface.co/aaronb/remove_anything).

- Gradio 

```bash
python -m remove_anything.app --ckpt checkpoints/big-lama.pt --device cuda
python -m remove_anything.app --model lama --device cuda
```

- Python API 

```python
from remove_anything import LaMa
model = LaMa(ckpt_path='big-lama.pt', device='cuda')
# image, mask is numpy array in the shape [H,W,3] and [H,W,1]
output = model(image, mask)
```

## Acknowledgement

[lama-cleaner](https://github.com/Sanster/lama-cleaner)