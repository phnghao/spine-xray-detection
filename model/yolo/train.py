from ultralytics import YOLO
import torch
import argparse

def train_yolo(data_yaml, epochs = 100, batch =4, imgsz = 640):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')

    # Huấn luyện YOLO + Tăng cường dữ liệu
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,    
        batch=batch,    
        device = device,    
       
        optimizer='SGD',
        lr0=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=5,

        mosaic=0.0,
        mixup=0.0,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,
        flipud=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        scale=0.05,
        translate=0.05,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        patience=30,
        close_mosaic=10,
        workers=8,
        cache=False
    )
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml-file', required = True, type = str)
    parser.add_argument('--epochs', type = int, default=100)
    parser.add_argument('--batch-size', type = int, default=4)
    parser.add_argument('--imgsz', type = int, default=640)

    args = parser.parse_args()

    train_yolo(args.yaml_file, epochs=args.epochs, batch = args.batch_size, imgsz = args.imgsz)

if __name__ == '__main__':
    main()


