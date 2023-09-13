# Remove Anything

<a href="https://huggingface.co/aaronb/remove_anything">
    <img alt="huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue">
</a>

A python package that includes some SOTA inpainting models that does not require text. 

- **Easy to use**: simple and unified API.
- **Self contained**: each method is self contained in a file or directory, easy to copy to use.


| Original | Mask | LaMa | LDM | MAT | 
| -- | -- | -- | --| --| 
|![](assets/lama.png) | ![](assets/lama.png) |![](assets/lama.png) |![](assets/lama.png) |![](assets/lama.png) |

## Install

```bash
pip install remove_anything
```

## Usage

Download checkpoint at [here](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt).

```python
from lama import LaMa
model = LaMa(ckpt_path='big-lama.pt', device='cuda')
# image, mask is numpy array
output = model(image, mask)
```

## Acknowledgement

[lama-cleaner](https://github.com/Sanster/lama-cleaner)