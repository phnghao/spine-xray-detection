@echo off

set ANNO_FILE=./data/raw/annotations/train.csv
set OUTPUT_DIR=./data/annotations_split/
set TRAIN_RATIO=0.85
set SEED=36

python preprocessing/annotation_split.py --anno-file %ANNO_FILE% --output-dir %OUTPUT_DIR% --train-ratio %TRAIN_RATIO% --seed %SEED%

pause
