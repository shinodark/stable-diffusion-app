import sys

import streamlit as st
import torch
from diffusers import (DDIMScheduler, LMSDiscreteScheduler,
                       StableDiffusionImg2ImgPipeline, StableDiffusionPipeline)

# Upscaler
sys.path.append("../Real-ESRGAN")
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from gfpgan import GFPGANer


def dummy_safety_checker(images, **kwargs):
    return images, False


lms = LMSDiscreteScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
)

last_pipe_type = None
pipe = None

def open_pipe(type="txt2img"):
    global last_pipe_type
    global pipe
    if last_pipe_type is not None and type == last_pipe_type:
        return pipe

    if type == "txt2img":
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", revision="fp16",
            torch_dtype=torch.float16, scheduler=lms
        ).to("cuda")

    elif type == "img2img":
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            scheduler=scheduler,
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

    else:
        print("Error, type have to be txt2img or img2img")
        return None

    pipe.enable_attention_slicing(slice_size = 4)
    pipe.safety_checker = dummy_safety_checker
    last_pipe_type = type
    return pipe


# Upscaler Model setup
def setup_upscaler(modelname: str, scale: float, half_precision: bool):
    if modelname == None or modelname == "RealESRGAN_x4plus":
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
    elif modelname in ["RealESRGAN_x4plus_anime_6B"]:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
    else:
        return None

    model_path = "../Real-ESRGAN/experiments/pretrained_models/{}.pth".format(modelname)
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
        gpu_id=0,
    )

    return upsampler
