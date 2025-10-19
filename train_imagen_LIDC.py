import argparse
import hashlib
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

    imagen = Imagen(
        unets = (unet1, unet2),
        channels=1,
        text_encoder_name = 't5-large',
        image_sizes = (128, 512),
        timesteps = 1000,
        cond_drop_prob = 0.1
    ).cuda()

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
        )   # 显式关掉分片shuffle
        # 不要调用 .decode(...)，避免触发内置的 json 解码器
        .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
        .map_tuple(
            load_img,          # real.png  -> PIL.Image
            load_img,          # gen.png   -> PIL.Image
            load_txt,          # prompt.txt-> str
            load_txt,          # huatuo.json(其实是纯文本)-> str
            lambda x: x        # __key__   -> str
        )
        .with_length(13953)
    )
    train_ds = base.select(lambda s: split_by_hash(s[-1], 20) != 0)
    val_ds = base.select(lambda s: split_by_hash(s[-1], 20) == 0)

    dataset_length = 13953
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
        split_valid_fraction = 0.0
    )


    trainer.load_from_checkpoint_folder()

    trainer.add_train_dataset(
        train_ds,
        batch_size=1,
        collate_fn=Collator(
            image_size=512,
            image_label='image',
            text_label='findings',
            name="google/t5-v1_1-large",
            channels='L',
            url_label=None
        )
    )
    trainer.add_valid_dataset(
        val_ds,
        batch_size=1,
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

            if not (step % 2000) and step > 0 and trainer.is_main:
                images = trainer.sample(texts=["The lungs are clear of focal consolidation, pleural effusion or pneumothorax. The heart size is normal. The mediastinal contours are normal. Multiple surgical clips project over the left breast, and old left rib fractures are noted."],
                                        batch_size=1, return_pil_images=True,
                                        stop_at_unet_number=unet)
                images[0].save(f'./samples/sample-{i}-{step // 2000}.png')
    trainer.save('./checkpoints/imagen.pt')

def train(args):
    unet_epochs = {
        1: 10,
        2: 20
    }
    for unet_number, epoch in unet_epochs.items():
        print('Training for unet number {}'.format(unet_number))
        train_one_unet(unet_number, epoch, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='./checkpoints')
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()