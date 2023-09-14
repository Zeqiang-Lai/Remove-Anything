import gradio as gr
import numpy as np
from PIL import Image

import remove_anything


def predict(inputs):
    image = inputs["image"].convert("RGB")
    mask = inputs["mask"].convert("L")

    image = np.array(image).astype('float32') / 255
    mask = np.array(mask).astype('float32')[:, :, None] / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    output = inpainter(image, mask)

    output = Image.fromarray(output)
    return output


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
    parser.add_argument(
        '--ckpt', type=str, default=None, help='Checkpoint Path'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lama',
        choices=remove_anything.available_models,
        help='Model type',
    )
    parser.add_argument('--port', type=int, default=10086, help='Port')
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='IP address')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    inpainter = remove_anything.load(args.model, args.ckpt, args.device)

    main(args)
