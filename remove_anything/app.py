import gradio as gr
import torch
import numpy as np
from PIL import Image
from inpainting_plugin import LDMInpainter, MATInpainter, FCFInpainter


ldm_inpainter = LDMInpainter(
    ckpt_path='checkpoints/ldm_inpainting_big.ckpt', 
    device=torch.device('cuda')
)

mat_inpainter = MATInpainter(
    resolution=512, 
    ckpt_path='checkpoints/mat_places_512.pth', 
    device=torch.device('cuda')
)

fcf_inpainter = FCFInpainter(
    ckpt_path='checkpoints/fcf_places_512.pth', 
    device=torch.device('cuda'),
    resolution=512, 
)

INPAINTERS = {
    'LDM': ldm_inpainter,
    'MAT': mat_inpainter,
    'FCF': fcf_inpainter,
}

from PIL import Image


def predict(model_type, inputs):
    inpainter = INPAINTERS[model_type]
    
    image = inputs["image"].convert("RGB")
    mask = inputs["mask"].convert("L")
    image_size = image.size
    
    image = np.array(image.resize((512, 512)))
    mask = np.array(mask.resize((512, 512)))
    
    output = inpainter(image, mask)
    
    output = Image.fromarray(output).resize(image_size)
    return output


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload").style(height=400)
            radio = gr.Radio(["FCF", "LDM", "MAT"], value="LDM", interactive=True)   
            btn = gr.Button("Run")
        with gr.Column():
            result = gr.Image(label="Result")
        btn.click(fn=predict, inputs=[radio, image], outputs=result)
        
demo.launch(server_name='0.0.0.0', server_port=10056)