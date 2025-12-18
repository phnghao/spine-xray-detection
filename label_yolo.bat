@echo off

:: =======================
:: Configurable variables
:: =======================
set META_DIR=./data/dataset_meta.csv
set CPUS=4

:: Train
set ANNO_TRAIN=./data/annotations_split/train.csv
set LABEL_TRAIN=./data/yolo/labels/train

:: Val
set ANNO_VAL=./data/annotations_split/val.csv
set LABEL_VAL=./data/yolo/labels/val

:: Test
set ANNO_TEST=./data/annotations_split/test.csv
set LABEL_TEST=./data/yolo/labels/test

:: =======================
:: Convert annotations to YOLO format
:: =======================
echo Converting train annotations...
python preprocessing/label_yolo.py --anno-dir %ANNO_TRAIN% --meta-dir %META_DIR% --output-dir %LABEL_TRAIN% --cpus %CPUS%

echo Converting val annotations...
python preprocessing/label_yolo.py --anno-dir %ANNO_VAL% --meta-dir %META_DIR% --output-dir %LABEL_VAL% --cpus %CPUS%

echo Converting test annotations...
python preprocessing/label_yolo.py --anno-dir %ANNO_TEST% --meta-dir %META_DIR% --output-dir %LABEL_TEST% --cpus %CPUS%

pause
