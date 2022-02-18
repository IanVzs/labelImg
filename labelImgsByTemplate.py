"""
标注图像

使用模版匹配为yolo做训练集
"""
import os
import aircv
from PyQt5 import QtGui

from libs.labelFile import LabelFile; lf = LabelFile()

uhome = os.environ['HOME']
TEMPLATE_FILE_PATH = os.path.join(uhome, "图片")
IMG_PATH = os.path.join(uhome, "图片")

def add_template(template_path, sub, tem):
    ui_path = os.path.join(template_path, sub)
    for i in os.listdir(ui_path):
        tem[i] = aircv.imread(os.path.join(ui_path, i))
    return tem
def get_template():
    tem = {}
    if not os.path.isdir(TEMPLATE_FILE_PATH):
        raise f"设置的模板地址错误, {TEMPLATE_FILE_PATH}不是一个文件目录"
    add_template(template_path=TEMPLATE_FILE_PATH, sub="", tem=tem)
    return tem

# 获取名字和坐标
dict_tempate = get_template()
def match_template_get_NP(img):
    dict_rst = {}
    for k, v in dict_tempate.items():
        if img.shape[0] <= v.shape[0] or img.shape[1] < v.shape[1]:
            continue
        rst = aircv.find_all_template(img, v, threshold=0.85)
        if rst:
            dict_rst[k] = rst
    return dict_rst

def trans2shapes(dict_rst, all_label):
    shapes = []
    for k, v_list in dict_rst.items():
        for v in v_list:
            label = k.replace(".png", "").replace(".jpg", "")
            if label not in all_label:
               all_label.append(label)
            single = {
                'label': label,
                'line_color': (123, 247, 68, 100),
                'fill_color': (123, 247, 68, 100),
                # 'points': [(221.02272727272728, 700.8181818181818), (356.0, 700.8181818181818), (356.0, 804.0), (221.02272727272728, 804.0)],
                'points': [v["rectangle"][0], v["rectangle"][2], v["rectangle"][3], v["rectangle"][1]],
                'difficult': False
            }
            shapes.append(single)
    return shapes, all_label
def label_img(dir_path):
    if not os.path.isdir(dir_path):
        return
    all_label = []
    for i in os.listdir(dir_path):
        if ("png" not in i) and ("jpg" not in i) and ("jpeg" not in i):
            continue
        img_path = os.path.join(dir_path, i)
        img = aircv.imread(img_path)
        import time
        a = time.time()
        dict_rst = match_template_get_NP(img)
        print("耗时: ", time.time() - a)
        filename = img_path.replace(".png", ".txt").replace(".jpg", ".txt")
        shapes, all_label = trans2shapes(dict_rst, all_label)
        qimage = QtGui.QImage(img_path)
        lf.save_yolo_format(
            filename=filename, shapes=shapes, image_path=img_path,
            image_data=qimage, class_list=all_label, line_color=(0, 255, 0, 128),
            fill_color=(255, 0, 0, 128)
        )
        print(f"{i} save success.")

if __name__ == "__main__":
    label_img(IMG_PATH)

