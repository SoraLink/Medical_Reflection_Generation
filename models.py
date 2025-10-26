import copy
import io
import os.path
import random
from pathlib import Path
from typing import List

import accelerate
import requests
import torch
from PIL import Image
from accelerate import Accelerator
from dalle_pytorch import VQGanVAE, DALLE
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, EMAModel, get_scheduler, \
    StableDiffusionControlNetPipeline
import torch.nn.functional as F
from imagen_pytorch import Unet, Imagen, ImagenTrainer, load_imagen_from_checkpoint, ElucidatedImagen
from dalle_pytorch.tokenizer import tokenizer
from packaging import version
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.nn as nn



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
    elif model == 'reflection':
        return Reflection()
    else:
        raise NotImplementedError

class Reflection:
    def __init__(self):
        deffusion_path = './output'
        self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            deffusion_path,
            torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False
        ).to("cuda:3")
        reflection_path = './controlnet'
        self.reflection_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            reflection_path,
            torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False
        ).to("cuda:1")
        self.huatuo = Huatuo()
        verifier_path = './bt_verifier_qwenvl/best_merged_1.0000'
        self.verifier = QwenVLVerifier(verifier_path)
        self.SYSTEM_PROMPT = (
            "You are a medical imaging evaluation assistant. "
            "You are given a generated chest X-ray image and its diagnostic description. "
            "Your task is to analyze the generated image against the description and provide concise reflection: "
            "what aspects should be improved to make the image 1) better match the diagnostic description, and "
            "2) look more realistic as a chest X-ray. "
            "Focus on anatomical accuracy, realism, and consistency. "
            "Keep the answer concise, no longer than 100 words."
        )

        self.USER_PROMPT = (
            "Diagnostic description:\n"
            "{diagnostic_description}\n\n"
            "Task:\n"
            "Provide short and clear suggestions to improve the generated image so it better reflects the description "
            "and looks more realistic, in under 100 words."
        )

    def __call__(self, prompts, num_inference_steps, num_reflection_steps=20):
        # 1) 初次生成
        generated_images = self.diffusion_pipe(
            prompt=prompts,
            num_inference_steps=num_inference_steps
        ).images  # List[PIL.Image]
        reflection_images = generated_images  # 当前候选
        best_scores = self.verifier.score(reflection_images, prompts)  # List[float]

        for step in range(num_reflection_steps):
            # 2) 为每张图生成一条 reflection 文本（作为下一轮的 prompt）
            print("Reflection step", step)
            reflections = []
            for idx, diag_prompt in enumerate(prompts):
                query = (self.SYSTEM_PROMPT + "\n" + self.USER_PROMPT).format(
                    diagnostic_description=diag_prompt
                )
                # 传入当前图 + 描述，让huatuo给出修改建议文本
                out_text = self.huatuo.inference(query, reflection_images[idx])
                reflections.append(out_text)

            def clip_trim(pipe, text: str, max_tokens: int = 75) -> str:
                tok = pipe.tokenizer
                ids = tok(text, truncation=True, max_length=max_tokens, add_special_tokens=True)["input_ids"]
                return tok.decode(ids, skip_special_tokens=True).strip()

            # 你的循环里这样用（没有 <|endoftext|> 清理步骤）
            reflections = [clip_trim(self.reflection_pipe, t, 75) for t in reflections]

            # 3) 用 reflection 文本 + 当前图 作为条件，生成下一轮图
            next_output = self.reflection_pipe(
                prompt=reflections,  # 新的文本（修改建议）
                image=reflection_images,  # 直接喂上一轮图做ControlNet条件
                num_inference_steps=num_inference_steps,
                controlnet_conditioning_scale=1.0,
                guidance_scale=7.5
            )
            next_reflection_images = next_output.images  # 取出图像列表

            # 4) 打分并逐个选择更优的图
            next_scores = self.verifier.score(next_reflection_images, prompts)  # List[float]

            updated_images = []
            for idx in range(len(prompts)):
                if next_scores[idx] > best_scores[idx]:
                    updated_images.append(next_reflection_images[idx])
                    best_scores[idx] = next_scores[idx]
                    print(True)
                else:
                    updated_images.append(reflection_images[idx])

            reflection_images = updated_images

        # 返回最终图、分数、最后一轮的reflection文本（可选）
        return reflection_images

class Huatuo:
    def __init__(self):
        self.server = "http://127.0.0.1:6006/inference"

    def inference(self, prompt: str, pil_image) -> str:
        if pil_image is None:
            raise ValueError("Pil image cannot be None")

        buf = io.BytesIO()
        pil_image.save(buf, "JPEG", quality=95)
        buf.seek(0)
        files = {'image': ("img.jpg", buf.getvalue(), "image/jpeg")}
        resp = requests.post(self.server, data={"prompt": prompt}, files=files, timeout=180)

        resp.raise_for_status()
        j = resp.json()
        if not j.get("ok", False):
            raise RuntimeError(j.get("error"))
        print(j['text'])
        return j['text']


class QwenVLVerifier:
    def __init__(self, ckpt_dir, device="cuda:2"):
        self.device = device

        base_dir = os.path.join(ckpt_dir, "base")
        proc_dir = os.path.join(ckpt_dir, "processor")
        head_path = os.path.join(ckpt_dir, "head.pt")

        # 加载已合并的基座
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_dir, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)

        # 加载标量头
        hidden = self.model.config.hidden_size
        self.head = ScalarHead(hidden).to(self.device)
        self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        self.head.eval()

    @torch.no_grad()
    def _score_batch(self, images, texts):
        """
        images: List[PIL.Image.Image 或 URL/base64（与训练一致）]
        texts:  List[str]
        return: Tensor [B]，每个样本的标量分数
        """
        # 构造与训练同样的 chat 格式
        text_inputs_batch, images_inputs_batch = [], []
        for i, img in enumerate(images):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": texts[i]},
                ],
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            image_inputs, _ = process_vision_info(messages)
            text_inputs_batch.append(text)
            images_inputs_batch.append(image_inputs)

        batch = self.processor(
            text=text_inputs_batch,
            images=images_inputs_batch,
            videos=None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)


        out = self.model(**batch, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]  # [B, T, H]
        B, T, H = hs.shape
        attn = batch.get("attention_mask", None)
        if attn is not None:
            idx = attn.to(torch.int64).sum(dim=-1) - 1
        elif "input_ids" in batch and getattr(self.model.config, "pad_token_id", None) is not None:
            pad_id = self.model.config.pad_token_id
            idx = (batch["input_ids"] != pad_id).to(torch.int64).sum(dim=-1) - 1
        else:
            idx = torch.full((B,), T - 1, device=hs.device, dtype=torch.long)
        idx = idx.clamp(min=0, max=T - 1)
        pooled = hs[torch.arange(B, device=hs.device), idx]  # [B, H]
        score = self.head(pooled)  # [B]
        return score

    # 单样本打分
    def score(self, images, texts):
        scores = self._score_batch(images, texts)
        return scores.detach().cpu().tolist()

        # 成对比较（如训练里的“win vs lose”）
    def compare(self, image_win, image_lose, prompt):
        sw, sl = self._score_batch([image_win, image_lose], [prompt, prompt])
        return {
            "score_win": float(sw.item()),
            "score_lose": float(sl.item()),
            "prefers_win": bool(sw.item() > sl.item()),
            "margin": float((sw - sl).item()),
        }

class ScalarHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, pooled):
        return self.mlp(pooled).squeeze(-1)

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
        print(f"Loading model from {model_path}")
        self.imagen = load_imagen_from_checkpoint(model_path).to(self.device)
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
        images = [img.convert("RGB") for img in images]
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
        text_tokens = tokenizer.tokenize(prompts, self.dalle.text_seq_len).cuda()
        output = self.dalle.generate_images(text_tokens)
        output = F.interpolate(output, size=(512, 512), mode="bilinear", align_corners=False)
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