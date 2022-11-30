import os
import os.path as osp
from glob import glob

cache_dir_path = "yolov5pip_ir_st"
# cache_dir_path = "yolov5pip_therm_st"
if cache_dir_path.find("ir") > 0:
    img_type = "ir"
else:
    img_type = "thermal"
train_path = osp.join(cache_dir_path, "data", "train")
val_path = osp.join(cache_dir_path, "data", "val")

yolo_dir_path = cache_dir_path.replace("_st", "_yolo")
os.makedirs(yolo_dir_path, exist_ok=True)
yolo_image_path = osp.join(yolo_dir_path, "images")
os.makedirs(yolo_image_path, exist_ok=True)
yolo_label_path = osp.join(yolo_dir_path, "labels")
os.makedirs(yolo_label_path, exist_ok=True)

train_txt_path = osp.join(yolo_dir_path, "train.txt")
val_txt_path = osp.join(yolo_dir_path, "val.txt")
test_txt_path = osp.join(yolo_dir_path, "test.txt")
import asyncio

import aiofiles.os as aos

tasks = []
for path in glob(osp.join(train_path, "*.jpg")):
    fn = os.path.basename(path)
    tasks.append(aos.rename(path, osp.join(yolo_image_path, fn)))
for path in glob(osp.join(val_path, "*.jpg")):
    fn = os.path.basename(path)
    tasks.append(aos.rename(path, osp.join(yolo_image_path, fn)))
for path in glob(osp.join(train_path, "*.txt")):
    fn = os.path.basename(path)
    tasks.append(aos.rename(path, osp.join(yolo_label_path, fn)))
for path in glob(osp.join(val_path, "*.txt")):
    fn = os.path.basename(path)
    tasks.append(aos.rename(path, osp.join(yolo_label_path, fn)))
assert len(tasks) > 0
print(f"Rename tasks: {len(tasks)}")
async def exec_tasks():
    await asyncio.gather(*tasks)
asyncio.run(exec_tasks())
yo_img_paths = glob(osp.join(yolo_image_path, "*.jpg"))
yo_label_paths = glob(osp.join(yolo_label_path, "*.txt"))
yo_img_paths = [osp.abspath(path) for path in yo_img_paths]
yo_label_paths = [osp.abspath(path) for path in yo_label_paths]
assert len(yo_img_paths) == len(yo_label_paths)
assert len(yo_img_paths) > 0
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(
    yo_img_paths,
    yo_label_paths,
    test_size=0.1,
    shuffle=True,
    stratify=None,
    random_state=34,
)
assert len([os.path.basename(row).replace(".jpg", ".txt") for row in x_train]) == len(
    [os.path.basename(row).replace(".txt", ".jpg") for row in y_train]
)
with open(train_txt_path, "w") as f:
    for row in x_train:
        f.write(row + "\n")
with open(val_txt_path, "w") as f:
    for row in x_valid:
        f.write(row + "\n")
import yaml

with open(osp.join(cache_dir_path, "data", "data.yml"), "r", errors="ignore") as f:
    cache_yaml = yaml.safe_load(f)
yolo_yaml = {}
yolo_yaml["train"] = osp.abspath(train_txt_path)
yolo_yaml["val"] = osp.abspath(val_txt_path)
yolo_yaml["names"] = cache_yaml["names"]
yolo_yaml["nc"] = cache_yaml["nc"]
with open(f"yolov5pip_{img_type}_yolo.yml", "w") as f:
    yaml.dump(yolo_yaml, f)
