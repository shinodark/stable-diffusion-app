import os
import torch
from diffusers import StableDiffusionPipeline

model = "runwayml/stable-diffusion-v1-5"
save_path = "D:/DEV/models"

model_path = os.path.join(save_path, model)
if not os.path.isdir(model_path):
    os.makedirs(model_path)

pipe = StableDiffusionPipeline.from_pretrained(
            model,
            revision="fp16",
            torch_dtype=torch.float16
        ).to("cuda")
pipe.save_pretrained(model_path)
