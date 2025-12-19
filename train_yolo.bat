@echo off

set YAML=./data/yolo/data.yaml
set EPOCHS=100
set IMGSZ=640

python -m model.yolo.train ^
    --yaml-file %YAML% ^
    --epochs %EPOCHS% ^
    --imgsz %IMGSZ%
pause