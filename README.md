# Remove Anything

[![](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Models-blue)](https://huggingface.co/aaronb/remove_anything )

A python package that includes some SOTA inpainting models that does not require text. 

- **Easy to use**: simple and unified API.
- **Self contained**: each method is self contained in a file or directory, easy to copy to use.


| Original                  | Mask                 |                      |                     |
| ------------------------- | -------------------- | -------------------- | ------------------- |
| ![](assets/original.jpeg) | ![](assets/mask.png) |                      |                     |
| MAT                       | FCT                  | LaMa                 | LDM                 |
| ![](assets/original.jpeg) | ![](assets/mask.png) | ![](assets/lama.png) | ![](assets/ldm.png) |

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
# image: (numpy array) [H,W,3] datarange=(0,1)
# mask: (numpy array) [H,W,1] datarange=(0,1)
output = model(image, mask)
```

## Acknowledgement

[lama-cleaner](https://github.com/Sanster/lama-cleaner)