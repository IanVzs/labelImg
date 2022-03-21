import os
import cv2
import base64
import requests
import numpy as np
from loguru import logger
from fastapi import FastAPI
from fastapi import FastAPI, Form

from model import Model

app = FastAPI()
model_26 = Model("26.pt")
model_26_back = Model("26_bak.pt")

sessionid = ".eJxVjktuwzAMRO_CdWxQiiRHWXbfMxgUKdpuCinwZ5Mgd09dZOPtvJmHecI2CVwhd4zsgzY-oDRO2TbkEzeC6KOlM1tycII6D1SmB61TLf39Bldzgp62dey3Jc_9v8oYOISJ-JbLTuSHylBbrmWdp9TulfZDl_a7Sv79-nQPgpGW8W-trktRQ3cJ-1dSzIkzYjDexGDRo0TrzsGGqELKSEjqTPS-E80XIXi9AUCpS1E:1nUkBG:byF5tpadXbDo6GCiHUdAwjKX4N3D2Ulrr2W01X4w3D4"
def get_token():
    res = requests.get('http://192.168.0.181:8080/api/current-user/token', cookies={'sessionid': sessionid})
    return res.json().get('token')
token = get_token()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

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
            ob_rst = model_26.run(img=img)
            ob_back_rst = model_26_back.run(img=img)
            ob_names = [i["name"] for i in ob_rst]
            ob_back_rst = [i for i in ob_back_rst if (i["name"] not in ob_names and i["name"] in ('dismiss', 'block_account', '1','2', '3', '4', '5'))]
            for v in ob_rst + ob_back_rst:
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
    # res["annotations"] = res["results"]
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5481)
