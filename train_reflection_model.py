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
    wds.WebDataset("/data/sora/Medical_Reflection_Generation/out/ds_huatuo-{000000..000010}.tar",
                   shardshuffle=False)
    .decode("pil")  # 只解码图片
    .rename(huatuo_txt="huatuo.json")  # 把 .json 改成 huatuo_txt
    .to_tuple("real.png", "gen.png", "prompt.txt", "huatuo_txt", "__key__")
    .map_tuple(lambda r, g, p, h, k: (r, g, p.decode("utf-8"), h.decode("utf-8"), k))
)

dataloader = DataLoader(dataset, batch_size=4, num_workers=2)

for real_img, gen_img, prompt_txt, huatuo_json, key in dataloader:
    # real_img / gen_img / merged_img 都是解码过的 Tensor (decode() 时转的)
    # prompt_txt 是 str，huatuo_json 是 dict
    print(key, prompt_txt)
    print(huatuo_json)
    break