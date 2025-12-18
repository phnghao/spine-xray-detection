@echo off

:: =======================
:: Configurable variables
:: =======================
set INPUT_DIR=./data/raw/dataset
set OUTPUT_FILE=./data/dataset_meta.csv
set CPUS=4

:: =======================
:: Convert DICOM metadata to CSV
:: =======================
echo Processing DICOM to CSV...
python preprocessing/metadata.py --input-dir %INPUT_DIR% --output-dir %OUTPUT_FILE% --cpus %CPUS%
:: python preprocessing/metadata.py --input-dir %INPUT_DIR% --output-dir %OUTPUT_FILE% --cpus %CPUS% --debug  :: uncomment if debug needed

pause
