@echo off
echo Created data.yaml

REM ====Create data.yaml for YOLO ====

set ROOT_DIR=./data/yolo
set TRAIN_IMAGES=images/train
set VAL_IMAGES=images/val
set OUTPUT=data.yaml

python -m model.yolo.make_yaml ^
    --root-dir %ROOT_DIR% ^
    --train-images %TRAIN_IMAGES% ^
    --val-images %VAL_IMAGES% ^
    --output %OUTPUT%

pause
