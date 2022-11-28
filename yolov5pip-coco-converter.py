import multiprocessing as mp
import os
import os.path as osp
import random
import shutil
from glob import glob

import ujson as json
from joblib import Parallel, delayed
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
from tqdm import tqdm

ts_dir = "/mnt/d/ts"
tl_dir = "/mnt/d/tl"
vs_dir = "/mnt/d/vs"
vl_dir = "/mnt/d/vl"
carmera_type_prefix = "IR"
# carmera_type_prefix = "THERM"
train_jsons = sorted(glob(tl_dir + f"/*_{carmera_type_prefix}_*/*/*"))
valid_jsons = sorted(glob(vl_dir + f"/*_{carmera_type_prefix}_*/*/*"))
train_imgs = sorted(glob(ts_dir + f"/*_{carmera_type_prefix}_*/*/*"))
valid_imgs = sorted(glob(vs_dir + f"/*_{carmera_type_prefix}_*/*/*"))
save_dir = f"/mnt/e/yolov5pip_{carmera_type_prefix.lower()}"
train_save_dir = osp.join(save_dir, "train")
valid_save_dir = osp.join(save_dir, "valid")
train_ann_path = osp.join(train_save_dir, "train.json")
valid_ann_path = osp.join(valid_save_dir, "valid.json")
os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(valid_save_dir, exist_ok=True)


def preprocess_filename(filename: str):
    return filename.replace("사건사고데이터이미지_", "")


category_map = {
    2: "자전거탄사람",
    3: "킥보드탄사람",
    0: "사람",
    6: "킥보드",
    5: "자전거",
    4: "오토바이",
    1: "오토바이탄사람",
}
category_converter = {
    "자전거탄사람": "bicycleman",
    "킥보드탄사람": "kickboardman",
    "사람": "man",
    "킥보드": "kickboard",
    "자전거": "bicycle",
    "오토바이": "motorcycle",
    "오토바이탄사람": "motorcycleman",
}


def convert_coco(img_path, save_dir, coco, is_train=True):
    img_save_path = osp.join(save_dir, preprocess_filename(osp.basename(img_path)))
    if is_train:
        label_path = img_path.replace("/ts/", "/tl/").replace(".jpg", ".json")
    else:
        label_path = img_path.replace("/vs/", "/vl/").replace(".jpg", ".json")
    try:
        info = json.load(open(label_path, "r"))
    except Exception as e:
        print(e)
    else:
        shutil.copy(img_path, img_save_path)
    img_base_filename = osp.basename(img_save_path)
    image_id = info["image"]["date_captured"]
    img = CocoImage(
        id=image_id,
        file_name=img_base_filename,
        height=info["image"]["size"]["height"],
        width=info["image"]["size"]["width"],
    )
    for ann in info["annotation"]:
        xmin = ann["bndbox"]["xmin"]
        xmax = ann["bndbox"]["xmax"]
        ymin = ann["bndbox"]["ymin"]
        ymax = ann["bndbox"]["ymax"]
        category_id = ann["property"]["category_id"]
        category_name = ann["property"]["name"]
        category_map[category_id] = category_name
        coco_ann = CocoAnnotation(
            bbox=[xmin, ymin, xmax - xmin, ymax - ymin],
            category_id=category_id,
            category_name=category_converter[category_name],
            image_id=image_id,
        )
        img.add_annotation(coco_ann)
    coco.add_image(img)


train_coco = Coco()
train_num = 100_000
train_ds = random.sample(train_imgs, train_num)
train_job = Parallel(n_jobs=mp.cpu_count() // 2, prefer="threads")(
    delayed(convert_coco)(img_path, train_save_dir, train_coco)
    for img_path in tqdm(train_ds)
)

valid_coco = Coco()
valid_num_start = 0
valid_num = int(len(valid_jsons) * 0.2)
valid_ds = valid_imgs[valid_num_start : valid_num_start + valid_num]
valid_job = Parallel(n_jobs=mp.cpu_count() // 2, prefer="threads")(
    delayed(convert_coco)(img_path, valid_save_dir, valid_coco, False)
    for img_path in tqdm(valid_ds)
)
for id, name in category_map.items():
    train_coco.add_category(CocoCategory(id=id, name=category_converter[name]))

for id, name in category_map.items():
    valid_coco.add_category(CocoCategory(id=id, name=category_converter[name]))

save_json(train_coco.json, train_ann_path)
save_json(valid_coco.json, valid_ann_path)
import os.path as osp

import yaml

yolo_yaml = {}
yolo_yaml["train_json_path"] = osp.abspath(train_ann_path)
yolo_yaml["train_image_dir"] = osp.abspath(train_save_dir)
yolo_yaml["val_json_path"] = osp.abspath(valid_ann_path)
yolo_yaml["val_image_dir"] = osp.abspath(valid_save_dir)
with open(f"yolov5pip_{carmera_type_prefix.lower()}_so.yml", "w") as f:
    yaml.dump(yolo_yaml, f)

print(train_coco.stats)
print(valid_coco.stats)
