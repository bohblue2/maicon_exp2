

# 국방AI경진대회 코드 사용법

- 최종 추론 결과 생성 스크립트: `track.sh`
- 최종 추론 결과: `Final_Submission.zip`
- RAW 데이터 경로는 다음과 같아야 합니다. (`/workspace/01_data/`) (초기 셋팅과 다름)
- 베이스라인 프로젝트 경로는 다음과 같아야 합니다. (`/workspace/02_baseline`) (초기 셋팅과 다름)
- Weight and Bias 와 굉장히 커플링 되어 있기에 네트워크 환경이 불안정 할 경우 학습 도중 에러가 발생할 수 있습니다.
- `Yolov5x`와 `StrongSORT`(`osnet_x0_25_market1501`, `osnet_x0_75_market1501`)를 사용하였습니다. `IR` 데이터와 `Thermal` 따로 분리하여 각각 모델을 학습시킨 모델을 사용하였습니다.
- 모델의 Detection 퍼포먼스를 분석하여 Tracking 단계에서 `confidence_thres`를 조정하여 일부 추가 점수를 얻었습니다.

#### 키워드

- Multiple Object Tracking
- Multiple Object Detection
- Mosaic Augmentation
- CSP-Darknet, Yolov5, StrongSORT

# 환경 설정

*최신 아나콘다 환경이 설치되어있어야합니다..(기준일: 2022-12-01)*

```bash

conda activate base
cd /workspace/Final_Submission
pip install ujson sahi aiofiles opencv-python wandb
cd /workspace/Final_Submission/Yolov5_StrongSORT_OSNet
pip install -r requirements.txt 
cd /workspace/Final_Submission/yolov5-pip
pip install ujson sahi aiofiles opencv-python wandb
pip install -r requirements.txt --use-feature=2020-resolver
```

# 핵심 파일 설명

- `/workspace/Final_Submission/yolov5-pip`: yolov5 프로젝트 파일
- `/workspace/Final_Submission/Yolov5_StrongSORT_OSNet`: Object Tracking 용 프로젝트
- `create_coco_format.sh` : COCO 데이터 Annotation 생성용 스크립트
- `create_yolo_format.sh`:  COCO데이터셋에서 YOLO 데이터셋으로 변환(yolov5 사용)
- `download_pretrained.sh`: 프리트레인된 모델 다운로드(*임의 계정으로 `wandb login <api_key>` 다운로드 가능*)
- `make_submission_zipfile.sh`: 최종 결과를 `/workspace/Final_Submission`폴더에 저장합니다.
- `track.sh`: 최종 결과를 생성하는 스크립트

# 전체 워크플로우 설명

```bash
create_coco_format.sh
create_yolo_format.sh
download_pretrained.sh
track.sh
make_submission_zipfile.sh
```

1. `/workspace` 폴더 구조를 아래와 같이 바꿉니다. (예선때 구조와 유사)
   1. `/workspace/01_data` : 서브 디렉토리 안에 train, test 데이터셋이 존재해야합니다.
   2. `/workpsace/02_baseline` : 서브 디렉토리 안에 baseline 프로젝트자 존재해야 합니다.
2. `create_coco_format.sh` 로 COCO Annotation을 `/workspace/01_data` 안에 생성합니다.
3. `create_yolo_format.sh`로 COCO Annotation을 Yolo 포맷으로 변경합니다. (`ultralytics` 의 Yolov5 사용하였음.)
4. `download_pretrained.sh`로 프리트레인된 모델을 다운로드 받습니다. 
   1. 해당 레포지토리는 (https://github.com/bohblue2/maicon_exp2) 에 공개되어 있습니다.
   2. 해당 모델의 프리트레인 모델은 `w_download_model.py`로 다운받을 수 있으며 해당 스크립트와 모델은 `wandb`사의 artifacts 를 이용해 11월 29일 최초 업로드 및 공개되었습니다. (https://wandb.ai/ryanbae/maicon_all)
   3. 해당 모델은 `wandb`서비스의 회원이라면 누구든지 `wandb api`를 통해 다운로드 받을 수 있으며, 문서 하단의 wandb link로도 접근할 수 있습니다.
5. `track.sh`로 준비된 프리트레인모델과 Yolo로 컨버트된 데이터셋으로 최종 결과물을 산출합니다.
6. `make_submission_zipfile.sh` 파일로 최종 결과물을 `/workspace/Final_Submission`으로 이동시킵니다.

------

# Original README.md Contents of this repository

# Infrared & Thermal Image Object Tracker with Yolov5 + StrongSORT with OSNet


## Introduction

This repository describes object tracking project on infrared and thermal images, provided in [AI-HUB](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=497). 7 different classes of objects - namely person, man on motorcycle, motorcycle, man on bicycle, bicycle, man on kickboard, and kickboard - are detected with [YOLOv5](https://github.com/ultralytics/yolov5). Detected results are then passed onto [StrongSORT](https://github.com/dyhBUPT/StrongSORT)[](https://arxiv.org/abs/2202.13514) tracker. In addition, [OSNet](https://github.com/KaiyangZhou/deep-person-reid)[](https://arxiv.org/abs/1905.00953), [OCSORT](https://github.com/noahcao/OC_SORT)[](https://arxiv.org/abs/2203.14360), and [ByteTrack](https://github.com/ifzhang/ByteTrack)[](https://arxiv.org/abs/2110.06864) are also available. This repository was designed and constructed based on the repository [YOLOv5 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet).


## Installation

```
cd <Yolov5_StrongSORT_OSNet>
pip install -r requirements.txt 
cd <yolov5-pip>
pip install -r requirements.txt
```

## Tracking

Further instruction regarding tracking parameters is available on the repository [YOLOv5 + StrongSORT with OSNet](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet).
```bash
$ python track.py
```

## Pretrained Model

Our pretrained models are available on wandb as the following. Note that we trained YOLOv5 separately on infrared images and thermal images, due to their innate qualitative differences.
|   Image set    |  Size   | Date of last update |                             Link                             |
| :------------: | :-----: | :-----------------: | :----------------------------------------------------------: |
| Infrared image | 691.2MB |    Nov 29, 2022     | [wandb](https://wandb.ai/ryanbae/maicon_all/artifacts/model/run_ir/v0/overview) |
| Thermal image  | 691.2MB |    Nov 29, 2022     | [wandb](https://wandb.ai/ryanbae/maicon_all/artifacts/model/run_thermal/v0/overview) |

Pretrianed models are also available through `w_download_model.py`. The code is as the following.

```bash
$ python w_download_model.py
```

```
import wandb
project = 'maicon_all'
entity = 'ryanbae'
run = wandb.init(project=project)
model = run.use_artifact(f'{entity}/{project}/run_ir:latest', type='model')
directory = model.download()
model = run.use_artifact(f'{entity}/{project}/run_thermal:latest', type='model')
directory = model.download()
```

Note that we trained our model on single NVIDIA A100 GPU(40GB).
