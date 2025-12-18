@echo off

set ANNO_FILE=./data/raw/annotations/dataset.csv
set OUTPUT_DIR=./data/annotations_split/
set TRAIN_RATIO=0.7
set VAL_RATIO=0.15
set TEST_RATIO=0.15
set SEED=36

python preprocessing/annotation_split.py --anno-file %ANNO_FILE% --output-dir %OUTPUT_DIR% --train-ratio %TRAIN_RATIO% --val-ratio %VAL_RATIO% --test-ratio %TEST_RATIO% --seed %SEED%

pause
