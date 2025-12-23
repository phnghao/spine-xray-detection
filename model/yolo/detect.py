from ultralytics import YOLO
from collections import defaultdict
import cv2 as cv
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_color():
    return {
        'Osteophytes': (0, 255, 0),            # Xanh lá
        'Disc space narrowing': (0, 0, 255),   # Xanh dương
        'Foraminal stenosis': (0, 255, 255),   # Xanh lơ (Cyan)
        'Spondylolysthesis': (255, 0, 255),    # Hồng cánh sen (Magenta)
        'Vertebral collapse': (255, 165, 0),   # Cam
        'Surgical implant': (255, 0, 0),       # Đỏ
        'Other lesions': (128, 0, 128)         # tím
    }

def draw_label(img, text, x, y, color):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thick = 2
    padding = 10

    (text_w, text_h), baseline = cv.getTextSize(text, font, font_scale, thick, )

    bg = [(x, y - text_h - padding * 2), (x + text_w + padding, y)]

    cv.rectangle(img, bg[0], bg[1], (0, 40, 80), -1)
    cv.rectangle(img, bg[0], bg[1], color, 2)
    cv.putText(img, text, (x + 5, y -padding), font, font_scale, color, thick, cv.LINE_AA)

def comparing(image_path, csv_path, image_id, model):
    # True image
    df = pd.read_csv(csv_path)
    boxes = df[(df['image_id'] == image_id) & (df['xmin'].notna())]

    if not os.path.exists(image_path):
        print(f'Image was not found')
        return
    
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_true = img.copy()
    img_pred = img.copy()
    
    for i, row in boxes.iterrows():

        label = row.lesion_type
        color = get_color().get(label, (255, 255, 255))

        x_min, y_min = int(row['xmin']), int(row['ymin'])
        x_max, y_max = int(row['xmax']), int(row['ymax'])
        label = row['lesion_type']

        cv.rectangle(img_true, (x_min, y_min), (x_max, y_max), color, 8)
        draw_label(img_true, f'{label}',x_min, y_min, color)
        
    # predicted image
    pred_model  = model(image_path)

    for box in pred_model[0].boxes:

        b = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = model.names[cls_id]

        color = get_color().get(label, (255, 255, 255))

        cv.rectangle(img_pred, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 8)
        draw_label(img_pred, f'{label} {conf:.2f}', int(b[0]), int(b[1]), color)
        
    plt.figure(figsize = (12, 7))
    plt.subplot(121)
    plt.imshow(img_true)
    plt.axis('off')
    plt.title('True Image')
    
    plt.subplot(122)
    plt.imshow(img_pred)
    plt.axis('off')
    plt.title('Predict Image (YOLO)')
    plt.tight_layout()
    plt.show()

def main():
    img_id = '00f61276be8d5f1067337de30bded315'
    img_path = "D:/Projects/spinexr_detection/data/images/test_images/00f61276be8d5f1067337de30bded315.png"
    csv_path = "D:/Projects/spinexr_detection/data/raw/annotations/test.csv"

    model = YOLO("D:/Projects/spinexr_detection/runs/detect/train/weights/best.pt")

    comparing(image_path = img_path, csv_path=csv_path, image_id=img_id, model = model)

if __name__ == '__main__':
    main()
    