#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author : Yingcheng Liu
# Email  : liuyc@mit.edu
# Date   : 09/02/2022
#
# Distributed under terms of the MIT license.

"""

"""
import argparse
import typing as tp
from contextlib import nullcontext
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from diffusers.pipelines.stable_diffusion.safety_checker import \
    StableDiffusionSafetyChecker
from joblib.memory import Memory
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import tqdm, trange
from transformers import AutoFeatureExtractor

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(
        numpy_to_pil(x_image), return_tensors="pt"
    )
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image, clip_input=safety_checker_input.pixel_values
    )
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def prepare_model(
    config_path: str, ckpt_path: str, cuda: bool, verbose=False
) -> torch.nn.Module:

    config = OmegaConf.load(config_path)

    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    device = torch.device("cuda" if cuda else "cpu")
    model = model.to(device)
    model.eval()

    return model


# NOTE(YL 09/02):: remember to clean up the cache if it gets too large
mem = Memory(location="/tmp/stable_diffusion")


@mem.cache(ignore=["model"])
def gen_images(
    model,
    prompt: str,
    batch_size: int,
    n_iter: int,
    H: int,
    W: int,
    C: int,
    f: float,
    scale: float,
    fixed_code: bool,
    plms: bool,
    cuda: str,
    seed: int,
) -> tp.List[Image.Image]:

    seed_everything(seed)

    device = torch.device("cuda" if cuda else "cpu")

    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    data = [batch_size * [prompt]]
    shape = [C, H // f, W // f]

    start_code = None
    if fixed_code:
        start_code = torch.randn([batch_size, *shape], device=device)

    gen_image_list = []
    for n in trange(n_iter, desc="Sampling"):
        for prompts in tqdm(data, desc="data"):

            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])

            c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                x_T=start_code,
            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

            for x_sample in x_checked_image:
                x_sample = 255.0 * x_sample
                gen_image_list.append(x_sample.astype(np.uint8))

    return gen_image_list


def main(opt):

    model = st.cache(prepare_model, allow_output_mutation=True)(
        f"{opt.config}", f"{opt.ckpt}", opt.cuda
    )

    with st.sidebar.form("Params"):
        prompt = st.text_area(
            "Prompt", "Boston is a beautiful, grand, historic, sensible city."
        )
        seed = st.number_input("Seed", min_value=0, max_value=1000000, value=42)

        H = st.number_input("H", min_value=64, max_value=1024, value=512)
        W = st.number_input("W", min_value=64, max_value=1024, value=512)

        C = st.number_input("C", min_value=1, max_value=32, value=4)
        f = st.number_input("f", min_value=1, max_value=16, value=8)
        scale = st.number_input("Scale", min_value=0.0, max_value=10.0, value=7.5)

        batch_size = st.number_input("Batch size", min_value=1, max_value=8, value=1)
        n_iter = st.number_input(
            "Number of iterations", min_value=1, max_value=100, value=2
        )
        fixed_code = st.checkbox("Fixed code", value=True)

        st.form_submit_button()

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
        gen_image_list = gen_images(
            model,
            prompt,
            batch_size,
            n_iter,
            H,
            W,
            C,
            f,
            scale,
            fixed_code,
            opt.plms,
            opt.cuda,
            seed,
        )

    for idx_img, img in enumerate(gen_image_list):
        img = Image.fromarray(img)
        st.image(img)

        buf = BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        file_name = f"{idx_img:04d}.png"
        btn = st.download_button(
            label="Download",
            data=byte_im,
            file_name=file_name,
            mime="image/png",
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action="store_true",
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="use cuda",
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"

    main(opt)
