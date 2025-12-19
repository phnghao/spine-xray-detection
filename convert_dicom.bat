@echo off

REM ===== Paths =====
set INPUT_DIR=./data/raw/dataset
set OUTPUT_DIR=./data/images/dataset
set CPUS=4
set LOG_FILE=./convert_train_log.txt
set DEBUG=--debug

REM ===== Create venv if not exists =====
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM ===== Activate venv =====
call venv\Scripts\activate


REM ===== Install dependencies (1st time safe) =====
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM ===== Run conversion =====
echo Converting DICOM images to PNG...
python preprocessing/dicom2png.py ^
  --input-dir %INPUT_DIR% ^
  --output-dir %OUTPUT_DIR% ^
  --cpus %CPUS% ^
  --log-file %LOG_FILE% ^
  %DEBUG%

pause
