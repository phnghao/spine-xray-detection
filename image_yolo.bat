@echo off

set ROOT_DIR=./data/images/train_images
set TRAIN_CSV=./data/annotations_split/train.csv
set VAL_CSV=./data/annotations_split/val.csv
set TRAIN_DIR=./data/yolo/images/train
set VAL_DIR=./data/yolo/images/val
set CPUS=8

echo Processing images for YOLO...
python preprocessing/image_yolo.py --root-dir %ROOT_DIR% --traincsv-file %TRAIN_CSV% --valcsv-file %VAL_CSV% --train-dir %TRAIN_DIR% --val-dir %VAL_DIR% --cpus %CPUS%

pause
