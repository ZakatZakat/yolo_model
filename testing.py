from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
model.train(data="my_custom_dataset.yaml", epochs=100, imgsz=640)