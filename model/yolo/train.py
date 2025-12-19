from ultralytics import YOLO
import torch
import argparse

def train_yolo(data_yaml):
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt')
    model.train(
        data = data_yaml ,
        epochs=100,
        imgsz=640,
        device = device, # 0 : GPU, else cpu 
    )
    return model

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--yaml-file', required = True, type = str)

    arg = parser.parse_args()

    train_yolo(arg.yaml_file)

if __name__ == '__main__':
    main()