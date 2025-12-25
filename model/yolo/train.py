from ultralytics import YOLO
import torch
import argparse

def train_yolo(data_yaml, epochs = 100, batch =4, imgsz = 640):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8s.pt')

    # Huấn luyện YOLO + Tăng cường dữ liệu
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,

        optimizer='AdamW',
        lr0=3e-4,
        weight_decay=1e-4,

        mosaic=0.5,
        mixup=0.0,
        fliplr=0.5,

        scale=0.1,
        translate=0.05,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        box=7.5,
        cls=0.5,
        dfl=1.5,
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


