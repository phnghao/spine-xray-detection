@echo off

set YAML=./data/yolo/data.yaml

python -m model.yolo.train ^
    --yaml-file %YAML%

pause