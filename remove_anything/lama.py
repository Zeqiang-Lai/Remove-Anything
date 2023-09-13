import torch
import numpy as np
from typing import Optional


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(
    img: np.ndarray,
    mod: int,
    square: bool = False,
    min_size: Optional[int] = None,
):
    """

    Args:
        img: [H, W, C]
        mod:
        square: 是否为正方形
        min_size:

    Returns:

    """
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    height, width = img.shape[:2]
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)

    if min_size is not None:
        assert min_size % mod == 0
        out_width = max(min_size, out_width)
        out_height = max(min_size, out_height)

    if square:
        max_size = max(out_height, out_width)
        out_height = max_size
        out_width = max_size

    return np.pad(
        img,
        ((0, out_height - height), (0, out_width - width), (0, 0)),
        mode="symmetric",
    )


class LaMa:
    name = "lama"
    pad_mod = 8
    pad_to_square = False
    min_size = None

    def __init__(self, ckpt_path, device):
        self.device = torch.device(device)
        self.model = (
            torch.jit.load(ckpt_path, map_location="cpu")
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def forward(self, image, mask):
        """Input image and output image have same size
        image: [H, W, C] RGB
        mask: [H, W]
        return: BGR IMAGE
        """

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        mask = (mask > 0) * 1
        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

        inpainted_image = self.model(image, mask)

        cur_res = inpainted_image[0].permute(1, 2, 0).detach().cpu().numpy()

        return cur_res

    def __call__(self, image, mask):
        origin_height, origin_width = image.shape[:2]
        pad_image = pad_img_to_modulo(
            image,
            mod=self.pad_mod,
            square=self.pad_to_square,
            min_size=self.min_size,
        )
        pad_mask = pad_img_to_modulo(
            mask,
            mod=self.pad_mod,
            square=self.pad_to_square,
            min_size=self.min_size,
        )

        result = self.forward(pad_image, pad_mask)
        result = result[0:origin_height, 0:origin_width, :]

        result = result * mask + image * (1 - mask)
        result = np.clip(result * 255, 0, 255).astype("uint8")
        return result
