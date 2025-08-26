import webdataset as wds
import io
import json

from PIL import Image
from torch.utils.data import DataLoader


def pil_loader(data):
    return Image.open(io.BytesIO(data)).convert("RGB")

def json_loader(data):
    return json.loads(data.decode("utf-8"))

def bytes_to_str(b):  # 用于 txt / “伪 json” 文本
    return b.decode("utf-8")
pattern = "/data/sora/Medical_Reflection_Generation/out/ds_huatuo-{000000..000010}.tar"
dataset = (
    wds.WebDataset(pattern, shardshuffle=False)
    .decode("pil")  # 解图片即可
    # 把 huatuo.json 重命名成 huatuo_txt（去掉 .json 扩展，避免被自动 json.loads）
    .rename(huatuo_txt="huatuo.json")
    .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo_txt", "__key__")
    .map_tuple(lambda x: x, lambda x: x, bytes_to_str, bytes_to_str, lambda x: x)
)

dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

for real_img, gen_img, prompt_txt, huatuo_json, key in dataloader:
    # real_img / gen_img / merged_img 都是解码过的 Tensor (decode() 时转的)
    # prompt_txt 是 str，huatuo_json 是 dict
    print(key, prompt_txt)
    print(huatuo_json)
    break