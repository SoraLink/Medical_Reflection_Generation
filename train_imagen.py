import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Collator
from imagen_pytorch.utils import safeget
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_unet(
    unet,
    epoches,
    args
):
    unet1 = Unet(
        dim = 512,
        cond_dim = 512,
        dim_mults = (1,2,3,4),
        num_resnet_blocks = 3,
        layer_attns = (False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        attn_heads=8
    )

    unet2 = Unet(
        dim = 128,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = (2, 4, 8, 8),
        layer_attns = (False, False, False, True),
        layer_cross_attns = (False, False, False, True),
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
        unets = (unet1, unet2, unet3),
        channels=1,
        text_encoder_name = 't5-large',
        image_sizes = (64, 256, 512),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()

    dataset = load_dataset("itsanmolgupta/mimic-cxr-dataset")
    dataset  = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset_length = len(dataset['train'])
    trainer = ImagenTrainer(
        imagen,
        use_ema=True,
        lr=1e-4,
        warmup_steps=500,
        checkpoint_path=args.output_path,
        checkpoint_every=200,
        cosine_decay_max_steps=dataset_length*epoches,
        max_grad_norm=1.0,
        split_valid_from_train=True,
        split_valid_fraction = 0.025
    )

    if args.is_load:
        trainer.load_from_checkpoint_folder()

    trainer.add_train_dataset(
        dataset['train'],
        batch_size=4,
        collate_fn=Collator(
            image_size=512,
            image_label='image',
            text_label='findings',
            name="google/t5-v1_1-large",
            channels='L',
            url_label=None
        )
    )



    for i in range(epoches):
        progress = tqdm(range(len(trainer.train_dl)))
        for step in progress:
            loss = trainer.train_step(unet_number=unet)
            progress.set_postfix(loss=f'{loss:.4f}')
            if not (step % 500) and step > 0 and trainer.is_main:
                valid_loss = trainer.valid_step(unet_number=unet)
                print(f'valid loss: {valid_loss}')

            if not (step % 200) and step > 0 and trainer.is_main:
                images = trainer.sample(texts=["The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips project over the left breast, and old left rib fractures are noted."],
                                        batch_size=1, return_pil_images=True,
                                        stop_at_unet_number=unet)
                images[0].save(f'./samples/sample-{i}-{step // 2000}.png')

def train(args):
    unet_epochs = {
        #1: 10,
        #2: 20,
        3: 20
    }
    for unet_number, epoch in unet_epochs.items():
        print('Training for unet number {}'.format(unet_number))
        train_one_unet(unet_number, epoch, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./checkpoints')
    parser.add_argument('--is_load', action='store_true')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()