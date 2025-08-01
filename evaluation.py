import argparse
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
    parser.add_argument('--model', type=str, choices=['minim'], default='minim')
    parser.add_argument('--modality', type=str, choices=['CXR', 'ChestCT'], default='CXR')
    parser.add_argument('--weight_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_dir', type=str, default='./data')
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()