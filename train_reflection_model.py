import argparse
import logging

import accelerate
import datasets
import diffusers
import torch
import transformers
import webdataset as wds
import io
import json

from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel, EMAModel
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.utils import ContextManagers

logger = get_logger(__name__, log_level="INFO")

def _bytes_to_pil(x):
    if isinstance(x, bytes):
        return Image.open(io.BytesIO(x)).convert("RGB")
    if isinstance(x, Image.Image):
        return x
    if torch.is_tensor(x):
        return T.ToPILImage()(x)
    raise TypeError(f"unexpected image type: {type(x)}")

_img_tf = T.Compose([
    T.Resize((512, 512)),   # 按需改尺寸
    T.ToTensor(),           # -> float32 [0,1], CxHxW
])

def load_img(x):
    return _img_tf(_bytes_to_pil(x))

def load_txt(x):
    return x if isinstance(x, str) else x.decode("utf-8")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="./")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", type=bool, default=True)
    return parser.parse_args()

class EMA:
    def __init__(self, params: List[torch.nn.Parameter], decay=0.9999):
        self.shadow = [p.detach().clone() for p in params]
        for p in self.shadow: p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, params):
        for s, p in zip(self.shadow, params):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)
    @torch.no_grad()
    def copy_to(self, params):
        for s, p in zip(self.shadow, params):
            p.data.copy_(s.data)

def main():
    args = parse_args()
    pattern = "/data/sora/Medical_Reflection_Generation/out/ds_huatuo-{000000..000010}.tar"

    dataset = (
        wds.WebDataset(pattern, shardshuffle=False)   # 显式关掉分片shuffle
        # 不要调用 .decode(...)，避免触发内置的 json 解码器
        .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
        .map_tuple(
            load_img,          # real.png  -> PIL.Image
            load_img,          # gen.png   -> PIL.Image
            load_txt,          # prompt.txt-> str
            load_txt,          # huatuo.json(其实是纯文本)-> str
            lambda x: x        # __key__   -> str
        )
    )

    def collate_fn(batch):
        reals, gens, prompts, huatuos, keys = zip(*batch)
        return {
            "real": torch.stack(reals, dim=0),   # [B,C,H,W]
            "gen":  torch.stack(gens,  dim=0),   # [B,C,H,W]
            "prompt": list(prompts),             # List[str]
            "reflection": list(huatuos),         # List[str]
            "key": list(keys),                   # List[str]
        }

    dataloader = DataLoader(dataset, batch_size=8, num_workers=2, collate_fn=collate_fn)

    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    controlnet = ControlNetModel.from_unet(unet)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.train()

    if args.use_ema:
        ema_controlnet = ControlNetModel.from_unet(unet)
        ema_controlnet = EMAModel(ema_controlnet.parameters(), model_cls=ControlNetModel, model_config=ema_controlnet.config)


    for batch in tqdm(dataloader):
        real_img = batch["real"]          # Tensor [B,C,H,W]
        gen_img  = batch["gen"]           # Tensor [B,C,H,W]
        prompt_txt = batch["prompt"]      # List[str]，长度 B
        huatuo_text = batch["reflection"] # List[str]
        key = batch["key"]                # List[str]



if __name__ == "__main__":
    main()