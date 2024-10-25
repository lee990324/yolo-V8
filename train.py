from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data="./data/acc10.yaml", epochs=500, patience=30)
# model.train(data="/home/bcml1/WBC project/Yolov5-Master/data/ETNET.yaml", epochs=100, patience=30, batch=32, imgsz=640)