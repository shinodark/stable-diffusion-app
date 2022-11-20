import sys
import os
import streamlit as st
import torch
from diffusers import (
    DDIMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)

# Upscaler
sys.path.append("../Real-ESRGAN")
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from gfpgan import GFPGANer

MODEL_PATH = "d:/dev/models"


def dummy_safety_checker(images, **kwargs):
    return images, False


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


class PipeManager:
    def __init__(self):
        self._type = None
        self._scheduler = None
        self._model = None
        self._dirty = True
        self._pipe = None

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, new_value):
        self._dirty = new_value != self._type
        self._type = new_value

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, new_value):
        self._dirty = new_value != self._scheduler
        self._scheduler = new_value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, new_value):
        self._dirty = new_value != self._model
        self._model = new_value

    def open_pipe(self):
        if self._dirty == False:
            return self._pipe

        # Scheduler creation
        if self._scheduler == "DDIM":
            scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif self._scheduler == "LMS":
            scheduler = LMSDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        elif self._scheduler == "Euler":
            scheduler = EulerDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )
        elif self._scheduler == "EulerA":
            scheduler = EulerAncestralDiscreteScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
            )

        # Pipe creation
        model_full_path = os.path.join(MODEL_PATH, self.model)
        if self._type == "txt2img":
            self._pipe = StableDiffusionPipeline.from_pretrained(
                model_full_path,
                revision="fp16",
                torch_dtype=torch.float16,
                scheduler=scheduler,
            ).to("cuda")

        elif self._type == "img2img":
            self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_full_path,
                scheduler=scheduler,
                revision="fp16",
                torch_dtype=torch.float16,
            ).to("cuda")

        else:
            print("Error, type have to be txt2img or img2img")
            return None

        self._pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_attention_slicing(slice_size=4)
        self._pipe.safety_checker = dummy_safety_checker
        self._dirty = False
        return self._pipe
