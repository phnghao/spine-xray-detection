from ultralytics import YOLO
import torch
import argparse

def train_yolo(data_yaml, epochs = 100, batch =4, imgsz = 640):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    model.train(
        data = data_yaml ,
        epochs=epochs,
        imgsz=imgsz,
        device = device, # 0 : GPU, else cpu 
        batch = batch
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