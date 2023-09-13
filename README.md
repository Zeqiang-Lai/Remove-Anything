# Remove Anything

<a href="https://huggingface.co/aaronb/remove_anything">
    <img alt="huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue">
</a>


![](demo.png)

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

