# huatuo_server.py
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import tempfile, shutil
from PIL import Image
import io
import base64

# 确保先能 import 到 HuatuoChatbot
# sys.path.append("/data/sora/HuatuoGPT-Vision")  # 如需
from cli import HuatuoChatbot

# 绑定单卡（按需设置）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

app = FastAPI(title="Huatuo Inference Service", version="0.1")

BOT = None

@app.on_event("startup")
def _init_bot():
    global BOT
    if BOT is None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        from cli import HuatuoChatbot
        BOT = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-34B")

def _bytes_to_tempfile(img_bytes) -> str:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    td = tempfile.mkdtemp()
    path = os.path.join(td, "img.jpg")
    img.save(path, "JPEG", quality=95)
    return path, td

@app.post("/inference")
async def inference(
    prompt: str = Form(..., description="文本prompt"),
    # 任选其一：上传图像文件或 base64
    image: Optional[UploadFile] = File(None, description="上传图片文件（可选）"),
    image_b64: Optional[str] = Form(None, description="base64图片字符串（可选，不要带data:前缀）"),
):
    tmp_dir = None
    try:
        image_paths: List[str] = []
        if image is not None:
            bytes_ = await image.read()
            path, tmp_dir = _bytes_to_tempfile(bytes_)
            image_paths.append(path)
        elif image_b64:
            bytes_ = base64.b64decode(image_b64)
            path, tmp_dir = _bytes_to_tempfile(bytes_)
            image_paths.append(path)

        # Huatuo 的接口是文本 + [图片路径列表]，没有图片也行
        out = BOT.inference(prompt, image_paths if image_paths else None)
        # 统一成 str（有的实现返回 list[str]）
        if isinstance(out, (list, tuple)):
            out_text = out[0]
        else:
            out_text = str(out)
        return {"ok": True, "text": out_text}
    except Exception as e:
        return JSONResponse({"ok": False, "error": repr(e)}, status_code=500)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    # 可换成 0.0.0.0:6006 供远程用；workers=1 确保独占GPU
    uvicorn.run(app, host="127.0.0.1", port=6006, reload=False, workers=1)
