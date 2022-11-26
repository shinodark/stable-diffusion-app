import os
import argparse
import torch
from diffusers import (
    StableDiffusionPipeline,
)

parser = argparse.ArgumentParser(prog="Stable Diffusion model download")
parser.add_argument("-id", "--model_id", required=True, type=str)
parser.add_argument(
    "-o", "--output_dir", required=False, type=str, default="d:/dev/models"
)
args = parser.parse_args()

pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id, torch_dtype=torch.float16
).to("cuda")

pipe.save_pretrained(os.path.join(args.output_dir, args.model_id.split("/")[-1]))
