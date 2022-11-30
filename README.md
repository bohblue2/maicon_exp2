# Infrared & Thermal Image Object Tracker with Yolov5 + StrongSORT with OSNet


## Introduction

This repository describes object tracking project on infrared and thermal images, provided in [AI-HUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=497). 7 different classes of objects - namely person, man on motorcycle, motorcycle, man on bicycle, bicycle, man on kickboard, and kickboard - are detected with [YOLOv5](https://github.com/ultralytics/yolov5). Detected results are then passed onto [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/abs/2202.13514) tracker. In addition, [OSNet](https://github.com/KaiyangZhou/deep-person-reid)[](https://arxiv.org/abs/1905.00953), [OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360), and [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864) are also available. This repository was designed and constructed based on the repository [YOLOv5 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet).


## Installation

```
git clone --recurse-submodules https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git  # clone recursively
cd Yolov5_StrongSORT_OSNet
pip install -r requirements.txt  # install dependencies
```

## Tracking

Further instruction regarding tracking parameters is available on the repository [YOLOv5 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet).
```bash
$ python track.py
```

## Pretrained Model

Our pretrained models are available on wandb as the following. Note that we trained YOLOv5 separately on infrared images and thermal images, due to their innate qualitative differences.
|  Image set   | Size  |Date of last update|                                          Link                                      |
|:------------:|:-----:|:-----------------:|:----------------------------------------------------------------------------------:|
|Infrared image|691.2MB|   Nov 29, 2022    |  [wandb](https://wandb.ai/ryanbae/maicon_all/artifacts/model/run_ir/v0/overview)   |
|Thermal image |691.2MB|   Nov 29, 2022    |[wandb](https://wandb.ai/ryanbae/maicon_all/artifacts/model/run_thermal/v0/overview)|

Pretrianed models are also available through 'w_download_model.py'. The code is as the following.
'''bash
$ python w_download_model.py
'''
'''
import wandb
project = 'maicon_all'
entity = 'ryanbae'
run = wandb.init(project=project)
model = run.use_artifact(f'{entity}/{project}/run_ir:latest', type='model')
directory = model.download()
model = run.use_artifact(f'{entity}/{project}/run_thermal:latest', type='model')
directory = model.download()
'''

Note that we trained our model on single NVIDIA A100 GPU(40GB).