from ultralytics import YOLO

model = YOLO('./runs/detect/train/weights/best.pt')
model.predict(source='/home/bcml1/WBC project/data/video/video3_grayscale.mp4', save=True)