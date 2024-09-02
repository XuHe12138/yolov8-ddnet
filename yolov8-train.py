from ultralytics import YOLO

model=YOLO("yolov8n.pt")

model.train(data='yolo-ddnet.yaml',workers=0,epochs=200,batch=16,patience=0)