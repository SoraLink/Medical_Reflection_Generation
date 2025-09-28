#!/usr/bin/env python3
import time
from pathlib import Path
from tcia_utils import nbia
from tqdm import tqdm

OUTDIR = Path("/data/lidc_series")
OUTDIR.mkdir(parents=True, exist_ok=True)

print("获取 series 列表（可能需要一会儿）...")
series_df = nbia.getSeries(collection="LIDC-IDRI")
# 确认列名
if "SeriesInstanceUID" not in series_df.columns:
    raise SystemExit("返回结果里没有 SeriesInstanceUID 列，接口可能变化")

series_uids = series_df["SeriesInstanceUID"].tolist()
print(f"Total series: {len(series_uids)}")

def download_one(uid, max_retry=5, sleep=10):
    for attempt in range(1, max_retry+1):
        try:
            print(f"Downloading {uid} (attempt {attempt})")
            nbia.downloadSeries(seriesUID=uid, path=str(OUTDIR))
            return True
        except Exception as e:
            print(f"  下载失败: {e}")
            time.sleep(sleep)
    return False

ok = 0
fail = 0
for uid in tqdm(series_uids, desc="Series"):
    success = download_one(uid)
    if success:
        ok += 1
    else:
        fail += 1

print(f"完成：成功 {ok}, 失败 {fail}. 存放在 {OUTDIR}")
