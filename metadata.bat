@echo off

set INPUT_DIR=./data/raw/train_images
set OUTPUT_FILE=./data/metadata.csv
set CPUS=4

:: =======================
:: Convert DICOM metadata to CSV
:: =======================
echo Processing DICOM to CSV...
python preprocessing/metadata.py --input-dir %INPUT_DIR% --output-dir %OUTPUT_FILE% --cpus %CPUS%

pause
