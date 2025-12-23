@echo off
REM =========================================
REM  DICOM to PNG preprocessing
REM =========================================

REM ===== Create venv if not exists =====
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM ===== Activate venv =====
call venv\Scripts\activate

REM ===== Install dependencies =====
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


REM =========================================
REM ========== TRAIN DATA ===================
REM =========================================

set INPUT_DIR=./data/raw/train_images
set OUTPUT_DIR=./data/images/train_images
set CPUS=4
set LOG_FILE=./convert_train_log.txt

set DEBUG=--debug

echo -----------------------------------------
echo Converting TRAIN DICOM to PNG...
echo -----------------------------------------

python preprocessing/dicom2png.py ^
  --input-dir "%INPUT_DIR%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --cpus %CPUS% ^
  --log-file "%LOG_FILE%" ^
  %DEBUG%


REM =========================================
REM ========== TEST DATA =====================
REM =========================================

set INPUT_DIR=./data/raw/test_images
set OUTPUT_DIR=./data/images/test_images
set CPUS=4
set LOG_FILE=./convert_test_log.txt

echo -----------------------------------------
echo Converting TEST DICOM to PNG...
echo -----------------------------------------

python preprocessing/dicom2png.py ^
  --input-dir "%INPUT_DIR%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --cpus %CPUS% ^
  --log-file "%LOG_FILE%" ^
  %DEBUG%

pause
