import gc
import random
import os

import numpy as np
import PIL
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from torch import autocast

from app_model import *

pipe_manager = PipeManager()

# Image preprocessing
def preprocess(image, sizemax):
    w, h = sizemax
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h))
    image.thumbnail(sizemax, resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


# Inference function
def infer(
    model: str,
    prompt: str,
    negative_prompt: str,
    init_image_file,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    scheduler: str,
    strength: float,
    num_images: int,
    seed: int,
    upsampling_model: str,
    upsampler_scale: float,
    upsampler_half_precision: bool,
    upsampler_gfpgan: bool,
    progress_bar,
):
    global pipe_manager

    if seed is None or seed == 0:
        seed = random.randrange(0, np.iinfo(np.uint32).max)
    generator = torch.Generator("cuda").manual_seed(seed)
    seed_origin = seed

    # pipe initialisation
    if init_image_file is not None:
        init_image = Image.open(init_image_file)
        init_image = preprocess(init_image, (width, height))
        pipe_manager.type = "img2img"
    else:
        pipe_manager.type = "txt2img"

    pipe_manager.scheduler = scheduler
    pipe_manager.model = model
    pipe = pipe_manager.open_pipe()

    upsampler_flag = False
    if upsampling_model != "None":
        upsampler = setup_upscaler(
            upsampling_model, upsampler_scale, upsampler_half_precision
        )
        if upsampler is not None:
            upsampler_flag = True

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    images_origin = []
    images_upscaled = []
    progress_total = num_images * (2 if upsampler_flag else 1)
    progress_count = 0
    with autocast("cuda"):
        for i in range(num_images):
            if pipe_manager.type == "txt2img":
                results_origin = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                    guidance_scale=cfg_scale,
                    generator=generator,
                ).images
            elif pipe_manager.type == "img2img":
                results_origin = pipe(
                    init_image=init_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    strength=strength,
                    generator=generator,
                ).images
            images_origin.extend(results_origin)
            progress_count = progress_count + len(results_origin)
            progress_bar.progress(float(progress_count / progress_total))

            if upsampler_flag:
                for im_origin in results_origin:
                    im_origin = np.asarray(im_origin)
                    if upsampler_gfpgan:
                        face_enhancer = GFPGANer(
                            model_path="../Real-ESRGAN/experiments/pretrained_models/GFPGANv1.3.pth",
                            upscale=upsampler_scale,
                            arch="clean",
                            channel_multiplier=2,
                            bg_upsampler=upsampler,
                        )
                        _, _, output = face_enhancer.enhance(
                            im_origin,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True,
                        )
                    else:
                        output, _ = upsampler.enhance(
                            im_origin, outscale=upsampler_scale
                        )
                    images_upscaled.append(output)
                    progress_count = progress_count + 1
                    progress_bar.progress(float(progress_count / progress_total))
            # deterministic seed for next image
            seed = seed + 1
            generator = torch.Generator("cuda").manual_seed(seed)
    return images_origin, images_upscaled, seed_origin


st.set_page_config(
    page_title="Stable Diffusion All in One",
    page_icon="????",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.subheader(
    "Stable Diffusion txt2img and img2img, with upscaling (RealESRGAN and GFPGAN)"
)
with st.sidebar:
    with st.expander("Optional image input"):
        init_image = st.file_uploader(
            "Initial Image", type=["png", "jpg", "webp", "jpeg"]
        )
        strength = st.slider(
            "Strength (used with initial image)",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=0.75,
        )
    width = st.slider("Width", min_value=256, max_value=768, step=64, value=512)
    height = st.slider("Height", min_value=256, max_value=768, step=64, value=512)
    steps = st.slider("Sampling Steps", min_value=1, max_value=150, step=1, value=30)
    cfg_scale = st.slider(
        "Classifier Free Guidance Scale",
        min_value=1.0,
        max_value=20.0,
        step=0.5,
        value=7.0,
    )
    scheduler = st.selectbox(
        "Scheduler",
        options=["DDIM", "LMS", "Euler", "EulerA"],
        index=1,
    )
    num_images = st.slider("Num Images", min_value=1, max_value=20, step=1, value=1)
    with st.expander("Upsampling"):
        upsampling_model = st.selectbox(
            "Upsampling Model",
            options=["None", "RealESRGAN_x4plus", "RealESRGAN_x4plus_anime_6B"],
            index=0,
        )
        if upsampling_model != "None":
            upsampler_scale = st.slider(
                "Upsampler scale", min_value=1.0, max_value=4.0, step=0.5, value=4.0
            )
            upsampler_half_precision = st.checkbox("Half precision", value=False)
            upsampler_gfpgan = st.checkbox(
                "Use GFPGAN for face enhancement", value=False
            )
        else:
            upsampler_scale = 0
            upsampler_half_precision = False
            upsampler_gfpgan = False
    seed = st.number_input("Seed", min_value=0)
    st.write(" ")
    st.write(" ")

# get models list
models_list = next(os.walk(MODEL_PATH))[1]
model = st.selectbox(
    "Model",
    options=models_list,
    index=0,
)
prompt = st.text_input(
    "Prompt",
    placeholder="bouquet of roses",
)
negative_prompt = st.text_input(
    "Negative prompt",
    placeholder="red rose",
)
with st.expander("Prompt build helper"):
    components.iframe(
        "https://promptomania.com/generic-prompt-builder/", height=600, scrolling=True
    )

if st.button("Run !"):
    with st.spinner("Generating..."):
        progress_bar = st.progress(0)
        prompt = prompt.replace(
            "!dream ", ""
        )  # in case of a copy / paste from a !dream prompt
        try:
            images_origin, images_upscaled, seed = infer(
                model,
                prompt,
                negative_prompt,
                init_image,
                width,
                height,
                steps,
                cfg_scale,
                scheduler,
                strength,
                num_images,
                seed,
                upsampling_model,
                upsampler_scale,
                upsampler_half_precision,
                upsampler_gfpgan,
                progress_bar,
            )
        except Exception as e:
            st.error(e, icon="????")
            raise e
        else:
            st.info("Seed : " + str(seed))
            if upsampling_model != "None":
                tab1, tab2 = st.columns(2)
                with tab1:
                    st.header("Original images")
                    with st.container():
                        st.image(images_origin)
                with tab2:
                    st.header("Upscaled images")
                    with st.container():
                        st.image(images_upscaled)
            else:
                with st.container():
                    st.image(images_origin)
