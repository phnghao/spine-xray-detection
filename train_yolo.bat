@echo off

set YAML=./data/yolo/data.yaml
set EPOCHS=180
set BATCH_SIZE=16
set IMGSZ=1024

python -m model.yolo.train ^
    --yaml-file %YAML% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --imgsz %IMGSZ%
pause