import webdataset as wds
import io
import json

from PIL import Image
from torch.utils.data import DataLoader


def load_img(b):
    return Image.open(io.BytesIO(b)).convert("RGB")

def load_txt(b):
    return b.decode("utf-8")

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
dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

for real_img, gen_img, prompt_txt, huatuo_json, key in dataloader:
    # real_img / gen_img / merged_img 都是解码过的 Tensor (decode() 时转的)
    # prompt_txt 是 str，huatuo_json 是 dict
    print(key, prompt_txt)
    print(huatuo_json)
    break