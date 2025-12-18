import pydicom
import argparse
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm
import numpy as np

def process_dicom_image(dicom_file):
    try:
        data = pydicom.dcmread(dicom_file)
    except Exception as error:
        return {}
    imageType = data.get("ImageType", "")
    imageType = filter(lambda e: e != '', imageType)
    imageType = '/'.join(imageType)

    return {
        'image_id': Path(dicom_file).stem,
        'image_height': data.get("Rows", 0),
        'image_width': data.get("Columns", 0),
        'patient_id': data.get('PatientID', ''),
        'study_id': data.get('StudyInstanceUID', ''),
    }

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required = True, type = str)
    parser.add_argument('--output-dir', required = True, type = str)
    parser.add_argument('--cpus', default = 2, type = int)
    parser.add_argument('--debug', action = 'store_true')

    args = parser.parse_args()
    indir = Path(args.input_dir)
    dicom_files = list(indir.glob('*dicom'))
    res = []

    if args.debug:
        dicom_files =dicom_files[:50]
    
    imple = Parallel(
        n_jobs = args.cpus,
        backend='multiprocessing',
        prefer= 'processes',
        verbose = 1
    )
    do = delayed(process_dicom_image)
    tasks= (do(dicom_file) for dicom_file in tqdm(dicom_files))
    res = imple(tasks)

    outfile = args.output_dir
    if args.debug:
        outfile = outfile.replace('csv', '.debug.csv')
    df = pd.DataFrame(res)
    df.to_csv(outfile, index = False)