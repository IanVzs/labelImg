import os
import sys
import torch
import numpy as np
from loguru import logger

from pathlib import Path
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_requirements, non_max_suppression, print_args, scale_coords)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


class Model:
    def __init__(self, pt_name):
        self.device = select_device('')
        self.model = self.load_model(pt_name)

    def load_model(self, pt_name):
        # Load model
        model = DetectMultiBackend(os.path.join(BASE_PATH, pt_name), device=self.device, dnn=False)
        self.stride, self.names, self.pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
        # Half
        self.half = False
        self.half &= (self.pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if self.pt or jit:
            model.model.half() if self.half else model.model.float()
        return model

    @torch.no_grad()
    def run(self, source=os.path.join(BASE_PATH, ''),  # 暂不支持
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
            dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.pt, img=img)
            bs = 1  # batch_size
        else:
            return {}
        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz), half=self.half)  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=augment, visualize=visualize)
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
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        rst.append({
                            "name": self.names[int(cls)],
                            "confidence": float(conf),
                            "result": [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[-1]) / 2)],
                            "xyxy": [int(i) for i in xyxy],
                        })

            logger.info(f'{s}Done. ({t3 - t2:.3f}s)')
        return rst
