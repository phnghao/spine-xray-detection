@echo off

set ROOT_DIR=./data/images/dataset
set TRAIN_CSV=./data/annotations_split/train.csv
set VAL_CSV=./data/annotations_split/val.csv
set TEST_CSV=./data/annotations_split/test.csv
set TRAIN_DIR=./data/yolo/images/train
set VAL_DIR=./data/yolo/images/val
set TEST_DIR=./data/yolo/images/test
set CPUS=8

echo Processing images for YOLO...
python preprocessing/image_yolo.py --root-dir %ROOT_DIR% --traincsv-file %TRAIN_CSV% --valcsv-file %VAL_CSV% --testcsv-file %TEST_CSV% --train-dir %TRAIN_DIR% --val-dir %VAL_DIR% --test-dir %TEST_DIR% --cpus %CPUS%

pause
