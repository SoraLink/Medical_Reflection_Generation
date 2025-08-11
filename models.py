import copy
import os.path
import random

import accelerate
import torch
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EMAModel, get_scheduler
import torch.nn.functional as F
from imagen_pytorch import Unet, Imagen, ImagenTrainer, load_imagen_from_checkpoint
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
    elif model == 'diffusion':
        return Diffusion(model_path, device)
    elif model == 'imagen':
        return ImagenModel(model_path, device)
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

class ImagenModel:

    def __init__(self, model_path, device):
        self.device = device
        unet1 = Unet(
            dim=512,
            cond_dim=512,
            dim_mults=(1, 2, 3, 4),
            num_resnet_blocks=3,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True),
            attn_heads=8
        )

        unet2 = Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=(False, False, False, True),
            layer_cross_attns=(False, False, False, True),
            attn_heads=8
        )

        unet3 = Unet(
            dim=128,
            dim_mults=(1, 2, 4, 8),
            num_resnet_blocks=(2, 4, 8, 8),
            layer_attns=False,
            layer_cross_attns=(False, False, False, True),
            attn_heads=8
        )

        imagen = Imagen(
            unets=(unet1, unet2, unet3),
            channels=1,
            text_encoder_name='t5-large',
            image_sizes=(64, 256, 512),
            timesteps=1000,
            cond_drop_prob=0.1
        ).cuda()
        try:
            self.imagen = load_imagen_from_checkpoint(model_path).to(self.device)
        except Exception:
            # 兜底：若没有打包的checkpoint，就用手动方式加载
            ckpt = torch.load(model_path, map_location=self.device)
            imagen.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
            self.imagen = imagen

        self.imagen.eval()

    @torch.no_grad()
    def __call__(self, prompts, num_inference_steps, stop_at_unet_number: int = 3, cond_scale: float = 3.0,
                 return_pil_images: bool = True):
        images = self.imagen.sample(
            texts=prompts,
            batch_size=len(prompts),
            return_pil_images=return_pil_images,
            stop_at_unet_number=stop_at_unet_number,
            cond_scale=cond_scale
        )
        return images

dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split='train')
sample = random.choice(dataset)
image = sample['image']
prompt = sample['findings']
diffusion = Diffusion('/data/sora/Medical_Reflection_Generation/checkpoints/MINIM/model', 'cuda')
pred = diffusion([prompt], num_inference_steps=500)
image.save("original.png")
pred[0].save("pred.png")