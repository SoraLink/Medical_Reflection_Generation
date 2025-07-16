import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from transformers import CLIPTokenizer

from metrics import Metrics
from models import build_model

DATASETS = {
    "CXR": "itsanmolgupta/mimic-cxr-dataset",
    "ChestCT": ""
}

PROMPT_FIELDS = {
    "CXR": ["findings"],
    "ChestCT": []
}

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def evaluate(args):
    model = build_model(args.modality, args.weight_path, args.model, args.device)
    dataset = load_dataset(DATASETS[args.modality], split='train')
    split_datasets = dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_datasets["test"]

    metrics = Metrics(args.device)
    to_tensor = ToTensor()

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
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    for batch_idx, (real_pils, prompts) in enumerate(loader):
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
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()