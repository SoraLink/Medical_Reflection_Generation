import copy
import os.path
import random
from pathlib import Path
from typing import List

import accelerate
import torch
from PIL import Image
from accelerate import Accelerator
from dalle_pytorch import VQGanVAE, DALLE
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EMAModel, get_scheduler
import torch.nn.functional as F
from imagen_pytorch import Unet, Imagen, ImagenTrainer, load_imagen_from_checkpoint, ElucidatedImagen
from dalle_pytorch.tokenizer import tokenizer
from packaging import version
from torchvision.transforms.functional import to_pil_image
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
    elif model == 'dalle':
        return DALLEModel(model_path, device)
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
            cond_drop_prob=0.1,
            timesteps=50
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

class DALLEModel:

    def __init__(self,  model_path, device):
        dalle_path = Path(model_path)
        load_obj = torch.load(str(dalle_path))
        dalle_params, vae_params, weights, vae_class_name, version = load_obj.pop('hparams'), load_obj.pop(
            'vae_params'), load_obj.pop('weights'), load_obj.pop('vae_class_name', None), load_obj.pop('version', None)
        vae = VQGanVAE()
        self.dalle = DALLE(vae=vae, **dalle_params).cuda()
        self.dalle.load_state_dict(weights)
        self.image_size = vae.image_size

    def __call__(self, prompts, num_inference_steps):
        text_tokens = self.dalle.generate_texts(prompts, self.dalle.text_seq_len).cuda()
        output = self.dalle.generate_images(text_tokens)
        pil_images = self.dalle_tensor_to_pil(output)
        return pil_images

    def dalle_tensor_to_pil(self, t: torch.Tensor) -> List[Image.Image]:
        """
        将 DALLE 输出张量转为 PIL.Image 列表。
        参数:
          t: torch.Tensor, [B,3,H,W] 或 [3,H,W]，float，通常在 [-1,1]
        返回:
          List[PIL.Image.Image]
        """
        if t is None:
            return []
        # 拉到 CPU、float32，兼容单张/批量
        t = t.detach().to("cpu", torch.float32)
        if t.dim() == 3:
            t = t.unsqueeze(0)

        # 映射到 [0,1]，适配扩散模型常见的 [-1,1] 输出
        if t.min() < 0:
            t = (t.clamp(-1, 1) + 1) / 2
        else:
            t = t.clamp(0, 1)

        return [to_pil_image(img) for img in t]

dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset", split='train')
sample = random.choice(dataset)
image = sample['image']
prompt = sample['findings']
diffusion = Diffusion('/data/sora/Medical_Reflection_Generation/checkpoints/MINIM/model', 'cuda')
pred = diffusion([prompt], num_inference_steps=500)
image.save("original.png")
pred[0].save("pred.png")