from ultralytics import YOLO
yolo=YOLO("runs/detect/train/weights/best.pt",task="detect")
result=yolo(source="screen",save=False,conf=0.3,show=False)