@echo off

set YAML=./data/yolo/data.yaml
set EPOCHS=100
set BATCH_SIZE=4
set IMGSZ=640

python -m model.yolo.train ^
    --yaml-file %YAML% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --imgsz %IMGSZ%
pause