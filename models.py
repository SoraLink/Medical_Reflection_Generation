import copy
import os.path
import random

import accelerate
import torch
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EMAModel, get_scheduler
import torch.nn.functional as F
from packaging import version
from tqdm import tqdm


class MINIM:
    MODAL_IDX = {
        'OCT': 1,
        'CXR': 2,
        'Fundus': 3,
        'BrainMRI': 4,
        'BreastMRI': 4,
        'ChestCT': 4
    }

    def __init__(self, modal, model_path, device):
        self.modal = modal
        modal_id = f'''{self.MODAL_IDX[modal]}'''
        unet = UNet2DConditionModel.from_pretrained(os.path.join(model_path, 'unets', modal_id, 'unet'))
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            unet=unet,
            safety_checker=None,
        ).to(device)

    def __call__(self, prompts, num_inference_steps):
        images = self.pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps
        ).images
        return images


def build_model(modal, model_path, model, device):
    if model == 'minim':
        return MINIM(modal, model_path, device)
    else:
        raise NotImplementedError

class Diffusion:

    def __init__(self,  model_path, device):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        ).to(device)

    def __call__(self, prompts, num_inference_steps):
        images = self.pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps
        ).images
        return images

dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split='train')
sample = random.choice(dataset)
image = sample['image']
prompt = sample['findings']
diffusion = Diffusion('/data/sora/Medical_Reflection_Generation/checkpoints/MINIM/model', 'cuda')
pred = diffusion([prompt], num_inference_steps=500)
image.save("original.png")
pred[0].save("pred.png")