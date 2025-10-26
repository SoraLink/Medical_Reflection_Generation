import argparse
import hashlib
import io
import os

import torch
import torch.nn.functional as F
import nibabel as nib
from PIL import Image

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
from transformers import CLIPTokenizer

from metrics import Metrics
from models import build_model

DATASETS = {
    "CXR": "itsanmolgupta/mimic-cxr-dataset",
    "ChestCT": "ibrahimhamamci/CT-RATE"
}

PROMPT_FIELDS = {
    "CXR": ["findings"],
    "ChestCT": ["Findings_EN"]
}

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def get_cxr_loader(args):
    dataset = load_dataset(DATASETS[args.modality], split='train')
    split_datasets = dataset.train_test_split(test_size=0.1, seed=42)
    test_dataset = split_datasets["test"]

    def collate_fn(batch):
        real_images = []
        prompts = []
        for item in batch:
            prompt = f'''{args.modality}:'''
            for field in PROMPT_FIELDS[args.modality]:
                prompt += f'''{item[field]}\n'''
            tokenized = tokenizer(prompt, truncation=False)
            if len(tokenized["input_ids"]) <= 77:
                real_images.append(item["image"].convert("RGB"))
                prompts.append(prompt)
        return real_images, prompts
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    return loader

def get_LIDC_loader(args):
    pattern = "/data/LIDC/augmented/ds_ct-{000000..000007}.tar"
    import torchvision.transforms as T
    def _bytes_to_pil(x):
        if isinstance(x, bytes):
            return Image.open(io.BytesIO(x)).convert("RGB")
        if isinstance(x, Image.Image):
            return x
        if torch.is_tensor(x):
            return T.ToPILImage()(x)
        raise TypeError(f"unexpected image type: {type(x)}")

    def load_img(x):
        return _bytes_to_pil(x)

    def load_txt(x):
        return x if isinstance(x, str) else x.decode("utf-8")

    def split_by_hash(key: str, m: int = 20):
        # 把样本稳定地分到 0..m-1 的桶里；m=20 相当于 5% 验证集
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % m
        return h

    import webdataset as wds

    base = (
        wds.WebDataset(
            pattern,
            nodesplitter=wds.split_by_node,
            shardshuffle=False
        )  # 显式关掉分片shuffle
        # 不要调用 .decode(...)，避免触发内置的 json 解码器
        .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
        .map_tuple(
            load_img,  # real.png  -> PIL.Image
            load_img,  # gen.png   -> PIL.Image
            load_txt,  # prompt.txt-> str
            load_txt,  # huatuo.json(其实是纯文本)-> str
            lambda x: x  # __key__   -> str
        )
    )
    val_ds = base.select(lambda s: split_by_hash(s[-1], 20) == 0)

    def collate_fn(examples):
        reals, gens, prompts, huatuos, keys = zip(*examples)
        pixel_values = [real.convert("L") for real in reals]

        texts = []
        for p in prompts:
            enc = tokenizer(p, truncation=True, max_length=75, add_special_tokens=True)
            p_trunc = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
            texts.append(p_trunc)

        return pixel_values, texts

    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=16,
    )
    return val_dataloader

def get_ct_rate_loader(args):
    dataset = load_dataset(
        DATASETS[args.modality],
        split='validation[:10]',
        name="reports",
        cache_dir=args.data_dir
    )
    def collate_fn(batch):
        real_images = []
        prompts = []
        for item in batch:
            prompt = f'''{args.modality}:'''
            for field in PROMPT_FIELDS[args.modality]:
                prompt += f'''{item[field]}\n'''
            tokenized = tokenizer(prompt, truncation=False)
            input_ids = tokenized["input_ids"]
            if len(input_ids) > 77:
                input_ids = input_ids[:77]
                prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
            prompts.append(prompt)
            name = item["VolumeName"]
            directory_name = "data_volumes/dataset/valid/"
            folder1 = name.split("_")[0]
            folder2 = name.split("_")[1]
            folder = folder1 + "_" + folder2
            folder3 = name.split("_")[2]
            subfolder = folder + "_" + folder3
            subfolder = directory_name + folder + "/" + subfolder
            path = subfolder + "/" + item["VolumeName"]
            vol = nib.load(path).get_fdata().astype("float32")
            mid_idx = vol.shape[2] // 2
            slice_2d = vol[:, :, mid_idx]
            image = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(image, size=(512, 512),
                                    mode='bilinear', align_corners=False)
            resized = resized.squeeze().cpu().numpy()
            normalized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            normalized = (normalized * 255).astype('uint8')
            pil_img = Image.fromarray(normalized).convert("RGB")

            real_images.append(pil_img)
        return real_images, prompts
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    return loader

def evaluate(args):
    model = build_model(args.modality, args.weight_path, args.model, args.device)
    metrics = Metrics(args.device)
    to_tensor = ToTensor()
    if args.modality == 'CXR':
        loader = get_cxr_loader(args)
    elif args.modality == 'ChestCT':
        loader = get_ct_rate_loader(args)
    elif args.modality == 'LIDC':
        loader = get_LIDC_loader(args)
    else:
        raise NotImplementedError

    for real_pils, prompts in tqdm(loader):
        gen_pils = model(
            prompts=prompts,
            num_inference_steps=100
        )

        real_t = torch.stack([to_tensor(img) for img in real_pils]).to(args.device)
        gen_t = torch.stack([to_tensor(img) for img in gen_pils]).to(args.device)
        metrics.add_images(real_t, gen_t)
    print(metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['minim', 'diffusion', 'imagen', 'dalle', 'reflection'], default='minim')
    parser.add_argument('--modality', type=str, choices=['CXR', 'ChestCT', 'LIDC'], default='LIDC')
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()