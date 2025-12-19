from ultralytics import YOLO

model = YOLO("D:/Projects/spinexr_detection/runs/detect/train/weights/best.pt")

results = model(
    'D:/Projects/spinexr_detection/data/yolo/images/test/8f971b60fd2dd0a5c802dd119585df20.png',
    conf=0.01,
    save=True,       # lưu ảnh
    save_txt=False,  # không lưu txt
    show=True      # không bật cửa sổ GUI
)

# kiểm tra kết quả
print(results[0].boxes.xyxy)  # bbox
print(results[0].boxes.cls)   # class id
print(results[0].boxes.conf)  # confidence
