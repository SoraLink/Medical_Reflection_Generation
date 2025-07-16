import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from metrics import Metrics
from models import build_model

DATASETS = {
    "CXR": "itsanmolgupta/mimic-cxr-dataset",
    "ChestCT": ""
}

PROMPT_FIELDS = {
    "CXR": ["findings", "impression"],
    "ChestCT": []
}

def evaluate(args):
    model = build_model(args.modality, args.weight_path, args.model, args.device)
    dataset = load_dataset(DATASETS[args.modality], split='train')
    print(f"Train dataset size: {len(dataset)} samples")
    metrics = Metrics(args.device)
    to_tensor = ToTensor()

    def collate_fn(batch):
        real_images = [item["image"].convert("RGB") for item in batch]
        prompts = []
        for item in batch:
            prompt = ""
            for field in PROMPT_FIELDS[args.modality]:
                prompt += f'''{field}: {item[field]}\n'''
            prompts.append(prompt)
        return real_images, prompts
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for batch_idx, (real_pils, prompts) in enumerate(loader):
        gen_pils  = model(
            prompt=prompts,
            num_inference_steps=100
        ).images
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
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()