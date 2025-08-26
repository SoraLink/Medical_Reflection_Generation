import webdataset as wds
import io
import json

from PIL import Image
from torch.utils.data import DataLoader


def pil_loader(data):
    return Image.open(io.BytesIO(data)).convert("RGB")

def json_loader(data):
    return json.loads(data.decode("utf-8"))

dataset = (
    wds.WebDataset("/data/sora/Medical_Reflection_Generation/out/ds_huatuo-{000000..000010}.tar")
    .decode()  # 自动解码常见格式（jpg, png, txt, json 等）
    .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo.json", "__key__")
)

dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

for real_img, gen_img, prompt_txt, huatuo_json, key in dataloader:
    # real_img / gen_img / merged_img 都是解码过的 Tensor (decode() 时转的)
    # prompt_txt 是 str，huatuo_json 是 dict
    print(key, prompt_txt)
    print(huatuo_json)
    break