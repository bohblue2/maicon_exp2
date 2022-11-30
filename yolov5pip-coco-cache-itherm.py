from sahi.utils.coco import export_coco_as_yolov5_via_yml

# yml_path = "./yolov5pip_ir_so.yml"  # data_coco_from yolov5-pip format
yml_path = "./yolov5pip_itherm_so.yml"  # data_coco_from yolov5-pip format
# yml_path = "./yolov5pip_all_so.yml"  # data_coco_from yolov5-pip format
# save_dir = "./yolov5pip_ir_st"
save_dir = "./yolov5pip_itherm_st"
# save_dir = "./yolov5pip_all_st"

data = export_coco_as_yolov5_via_yml(
    yml_path=yml_path, output_dir=save_dir + "/" + "data"
)
import os
from pathlib import Path
from shutil import copyfile

import yaml

# add coco fields to data.yaml
with open(yml_path, errors="ignore") as f:
    data_info = yaml.safe_load(f)  # load data dict
with open(data, errors="ignore") as f:
    updated_data_info = yaml.safe_load(f)  # load data dict
    updated_data_info["train_json_path"] = data_info["train_json_path"]
    updated_data_info["val_json_path"] = data_info["val_json_path"]
    updated_data_info["train_image_dir"] = data_info["train_image_dir"]
    updated_data_info["val_image_dir"] = data_info["val_image_dir"]
    if data_info.get("yolo_s3_data_dir") is not None:
        updated_data_info["yolo_s3_data_dir"] = data_info["yolo_s3_data_dir"]
    if data_info.get("coco_s3_data_dir") is not None:
        updated_data_info["coco_s3_data_dir"] = data_info["coco_s3_data_dir"]

    updated_data_info["names"] = [
        "man",
        # "motorcycleman",
        # "bicycleman",
        # "kickboardman",
        # "motorcycle",
        # "bicycle",
        # "kickboard",
    ]
    print(f"Force Updating: {updated_data_info['names']}")
with open(data, "w") as f:
    yaml.dump(updated_data_info, f)

w = save_dir + "/" + "data" + "/" + "coco"  # coco dir
os.makedirs(w, exist_ok=True)  # make dir

# copy train.json/val.json and coco_data.yml into data/coco/ folder
if "train_json_path" in data_info and Path(data_info["train_json_path"]).is_file():
    copyfile(data_info["train_json_path"], str(w + "/" + "train.json"))
if "val_json_path" in data_info and Path(data_info["val_json_path"]).is_file():
    copyfile(data_info["val_json_path"], str(w + "/" + "val.json"))
