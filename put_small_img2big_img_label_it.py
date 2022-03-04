"""
目录结构:
├── ready_train_img.py # 脚本
├── big # 大图集合
│   └── 20220304-170511.jpeg
├── small # 小图集合
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   └── 5
├── suture # 导出集合
│   └── 20220304-170511.jpeg

运行:
python ready_train_img.py --img_path "./big" --simg_path "./small"

test某方法输出:
python ready_train_img.py --test get_left_up_points
"""
import os
import cv2
import random
import argparse

from labelImgsByTemplate import trans2shapes, save_yolo_fromat

def get_left_up_points(big_shape, small_shape, num=1):
    """
    网格贴图, 先划分网格再进行随机获取
    从一个大图中随机获取可以贴小图的位置
    ret: 小图左上角点合集
    """
    points = []
    if len(big_shape) < 2 or len(small_shape) < 2:
        return points
    bx, by = big_shape[:2]
    sx, sy = small_shape[:2]
    x_times = bx // sx
    y_times = by // sy

    xs = [ i*sx for i in range(x_times)]
    ys = [ i*sy for i in range(y_times)]
    for x in xs:
        for y in ys:
            # xy 用shape读取出来的是反的, 这里矫正
            points.append((y, x))
    if len(points) <= num:
        return points
    else:
        ps = set([])
        rate = num/len(points)
        while len(ps) < num:
            for p in points:
                if random.random() < rate:
                    ps = ps | {p, }
        points = list(ps)[:num]
    import ipdb; ipdb.set_trace()
    return points
    

def put_img2img(small, big, left_up_x, left_up_y):
    """
    samll + big ==> new
        111     111
    2 + 111 ==> 121
        111     111
    """
    smallB = big - big
    smallB[left_up_y:left_up_y+small.shape[0],left_up_x:left_up_x+small.shape[1]] = small
    # return cv2.addWeighted(big, 1, smallB, 1, 0)
    big[left_up_y:left_up_y+small.shape[0],left_up_x:left_up_x+small.shape[1]] = small
    return big


def chg_img_name2txt_name(img_name):
    return img_name.replace(".png", '.txt').replace(".jpg", '.txt').replace(".jpeg", '.txt')

def getimg_from_dir(dir_path):
    """
    传入文件夹, 使用cv2读取文件夹下所有图片
    """
    dict_img = {}
    if os.path.isdir(dir_path):
        for ff in os.listdir(dir_path):
            if ".png" in ff or ".jpg" in ff or ".jpeg" in ff:
                img = cv2.imread(os.path.join(dir_path, ff))
                dict_img[ff] = img
            else:
                continue
    return dict_img

def run(big_path, small_path_father):
    labels = {}
    dict_big = getimg_from_dir(big_path)
    max_x = 0
    max_y = 0

    for big_name, big_img in dict_big.items():
        dict_small = {}
        labels[big_name] = {}
        for dir_small in os.listdir(small_path_father):
            if dir_small == ".DS_Store":
                # 垃圾 macos
                continue
            _dict_small = getimg_from_dir(os.path.join(small_path_father, dir_small))
            list_s_keys = list(_dict_small.keys())
            rand_choise = random.randint(0, len(list_s_keys)-1)
            _small_name = list_s_keys[rand_choise]
            img_choise_you = _dict_small[_small_name]

            max_x = img_choise_you.shape[0] if max_x < img_choise_you.shape[0] else max_x
            max_y = img_choise_you.shape[1] if max_y < img_choise_you.shape[1] else max_y
            
            dict_small[dir_small] = img_choise_you
    
        list_small_name = list(dict_small.keys())
        list_left_up_points = get_left_up_points(big_img.shape, (max_x, max_y), num=len(list_small_name))
        
        put_num = 0
        for left_up_x, left_up_y in list_left_up_points:
            small_name = list_small_name[put_num]
            small_img = dict_small[small_name]
            right_down_x, right_down_y = left_up_x+small_img.shape[0], left_up_y+small_img.shape[1]
            big_img = put_img2img(small_img, big_img, left_up_x, left_up_y)            
            # print("nani????", left_up_x, left_up_y, right_down_x, right_down_y, "\t\t\t", small_img.shape)
            # cv2.imshow("", big_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            labels[big_name][small_name] = [(left_up_x, left_up_y), (right_down_x, right_down_y)]
            put_num += 1
        
        # save chged big img && positions
        cv2.imwrite(os.path.join("suture", big_name), big_img)
        big_img_path = os.path.join(big_path, big_name)
        save_label(big_img_path, labels[big_name])
        print(labels[big_name])

def save_label(img_path, label_info):
    all_label = ["1", "2", "3", "4", "5"]
    filename = chg_img_name2txt_name(img_path)
    # TODO BUG POINT
    filename = filename.replace("big", "suture")
    dict_rst = {}
    for label_name, v in label_info.items():
        (left_up_x, left_up_y), (right_down_x, right_down_y) = v
        dict_rst[label_name] = [
            # [(左上), (左下), (右上), (右下)]
            {"rectangle": [(left_up_x, left_up_y), (left_up_x, right_down_y), (right_down_x, left_up_y), (right_down_x, right_down_y)]}
        ]
    shapes, all_label = trans2shapes(dict_rst, all_label)
    save_yolo_fromat(filename, shapes, img_path, all_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, help="test func name")
    parser.add_argument("--img_path", type=str, help="big img path")
    parser.add_argument("--simg_path", type=str, help="small img path")

    args = parser.parse_args()
    PATH = args.img_path
    sPATH = args.simg_path

    dict_func = {
        "run": {"func": run, "args": {"big_path": "./", "small_path": ""}},
        "get_left_up_points": {
            "func": get_left_up_points,
            "args": {"big_shape": (100, 100), "small_shape": (10, 10),"num": 1}
        },
    }
    dict_func = dict_func.get(args.test)

    if dict_func:
        ret = dict_func["func"](**dict_func["args"])
        print(f"ret:\n {ret}")
    if PATH and os.path.exists(PATH) and PATH and os.path.exists(PATH):
        run(PATH, sPATH)
    else:
        print(f"输入合法路径")
