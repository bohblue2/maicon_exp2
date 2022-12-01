# Create YOLO Dataset(raw dataset => coco(cache) => yolo)
cd /workspace/Final_Submission
python /workspace/Final_Submission/yolov5pip-coco-cache-ir.py
python /workspace/Final_Submission/yolov5pip-coco-cache-therm.py
python /workspace/Final_Submission/yolov5pip-yolo-ir.py
python /workspace/Final_Submission/yolov5pip-yolo-therm.py
