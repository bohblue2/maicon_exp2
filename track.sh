which python


cd /workspace/Final_Submission/Yolov5_StrongSORT_OSNet

python track.py --source "/workspace/01_data/test/THERM_*" \
--yolo-weights /workspace/Final_Submission/yolov5-pip/weights/thermal/best.pt \
--reid-weights osnet_x0_25_market1501.pt \
--classes 0 \
--device 0 \
--name smurfs \
--project submission \
--save-txt \
--exist-ok 

python track.py --source "/workspace/01_data/test/IR_*" \
--yolo-weights /workspace/Final_Submission/yolov5-pip/weights/ir/best.pt \
--reid-weights osnet_x0_75_market1501.pt \
--classes 0 \
--conf-thres 0.75 \
--iou-thres 0.50 \
--name smurfs \
--project submission \
--device 0 \
--save-txt \
--exist-ok

cd /workspace/Final_Submission