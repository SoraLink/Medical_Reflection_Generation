import torch
import webdataset as wds
import io
import json

from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T

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

for batch in dataloader:
    real_img = batch["real"]          # Tensor [B,C,H,W]
    gen_img  = batch["gen"]           # Tensor [B,C,H,W]
    prompt_txt = batch["prompt"]      # List[str]，长度 B
    huatuo_text = batch["reflection"] # List[str]
    key = batch["key"]                # List[str]

    print(key[0], prompt_txt[0])
    print(huatuo_text[0])
    break