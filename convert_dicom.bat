@echo off

set INPUT_DIR=./data/raw/dataset
set OUTPUT_DIR=./data/images/dataset
set CPUS=4
set LOG_FILE=./convert_train_log.txt
::set DEBUG=--debug  


echo Converting DICOM images to PNG...
python preprocessing/dicom2png.py --input-dir %INPUT_DIR% --output-dir %OUTPUT_DIR% --cpus %CPUS% --log-file %LOG_FILE% %DEBUG%

pause
