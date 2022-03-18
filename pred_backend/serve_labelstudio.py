import os
import cv2
import torch
import base64
import requests
import numpy as np
from pathlib import Path
from loguru import logger
from fastapi import FastAPI
from fastapi import FastAPI, Form
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords)

app = FastAPI()

model = None

sessionid = ".eJxVjktuwzAMRO_CdWxQiiRHWXbfMxgUKdpuCinwZ5Mgd09dZOPtvJmHecI2CVwhd4zsgzY-oDRO2TbkEzeC6KOlM1tycII6D1SmB61TLf39Bldzgp62dey3Jc_9v8oYOISJ-JbLTuSHylBbrmWdp9TulfZDl_a7Sv79-nQPgpGW8W-trktRQ3cJ-1dSzIkzYjDexGDRo0TrzsGGqELKSEjqTPS-E80XIXi9AUCpS1E:1nUkBG:byF5tpadXbDo6GCiHUdAwjKX4N3D2Ulrr2W01X4w3D4"
def get_token():
    res = requests.get('http://192.168.0.181:8080/api/current-user/token', cookies={'sessionid': sessionid})
    return res.json().get('token')
token = get_token()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
import sys

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync

# Load model
device = select_device('')
try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    BASE_PATH = sys._MEIPASS
    ROOT = sys._MEIPASS
except Exception:
    pass
model = DetectMultiBackend(os.path.join(BASE_PATH, "26.pt"), device=device, dnn=False)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size((640, 640), s=stride)  # check image size
# Half
half = False
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()

def convert_to_ls(x, y, width, height, original_width, original_height):
    return x / original_width * 100.0, y / original_height * 100.0, \
           width / original_width * 100.0, height / original_height * 100

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

@app.get("/health")
async def health():
    return {"msg": "Hi"}

@app.post("/setup")
async def setup(data: dict):
    project = int(float(data["project"]))
    if not os.path.exists(f"{project}.pt"):
        # TODO 报错
        return {}
    return {}

def get_img_by_url(uri):
    url = f"http://192.168.0.181:8080{uri}"
    res = requests.get(url, headers= {'Authorization': f'Token {token}'}, cookies={'sessionid': sessionid})
    b64_byte = res.content
    if res.status_code!= 200:
        return np.ndarray(0)
    fdata = base64.b64decode(str(base64.b64encode(b64_byte))[2:])
    ndata = np.asarray(bytearray(fdata), dtype="uint8")
    img = cv2.imdecode(ndata, cv2.IMREAD_COLOR)
    return img

@torch.no_grad()
def ob_run(
        source=os.path.join(BASE_PATH, ''),  # 暂不支持
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        img=None,  # 直接传入img
    ):
    rst = []
    is_img = isinstance(img, np.ndarray)
    # Dataloader
    if is_img:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, img=img)
        bs = 1  # batch_size
    else:
        return {}
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    rst.append({
                        "name": names[int(cls)],
                        "confidence": float(conf),
                        "result": [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[-1]) / 2)],
                        "xyxy": [int(i) for i in xyxy],
                    })

        logger.info(f'{s}Done. ({t3 - t2:.3f}s)')
    return rst

@app.post("/predict")
def predict(data: dict):
    dict_label = {
        "defeat_1": "defeat",
        "defeat_2": "defeat",
        "draw2": "draw",
        "draw3": "draw",
        "vict": "victory"
    }
    res =  {"results": [{ "result": []}]}
    all_rst = []
    for task in data["tasks"]:
        img_path = task["data"]["image"]
        logger.debug(img_path)
        img = get_img_by_url(img_path)
        null_label = {
            "original_width": 1600,
            "original_height": 900,
            "image_rotation": 0,
            "value": {
            },
            # "from_name": "label",
            # "to_name": "image",
            # "type": "rectanglelabels",
            # "score": 100
        }
        if isinstance(img, np.ndarray) and img.any():
            height, width, _ = img.shape
            ob_rst = ob_run(img=img)
            for v in ob_rst:
                # result = v["result"] # 中心位置
                result = (v["xyxy"][0], v["xyxy"][1]) # 左上角
                xyxy = v["xyxy"]
                label = v["name"]
                score = v["confidence"] * 100
                if dict_label.get(label):
                    label = dict_label.get(label)
                x, y, w, h = convert_to_ls(result[0], result[-1], abs(xyxy[2]-xyxy[0]), abs(xyxy[3]-xyxy[1]), width, height)
                rst = {
                    "original_width": 1600,
                    "original_height": 900,
                    "image_rotation": 0,
                    "value": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "rotation": 0,
                        "rectanglelabels": [
                            label
                        ]
                    },
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "score": score
                }
                if label not in ("adventure", "back", "block_account", "bloodmoon", "closestool_left", "closestool_right", "close_down", "defeat", "disconnect", "dismiss", "draw", "endturn", "enermy_info", "maintenance",
"ok", "pvp", "rating", "retry", "round", "start", "team", "update", "victory", "vs", "wait_net", "cutscenes"):
                    all_rst.append(null_label)
                else:
                    all_rst.append(rst)
        else:
            all_rst.append(null_label)
    res["results"][0]["result"] = all_rst
    res["annotations"] = res["results"]
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5481)
