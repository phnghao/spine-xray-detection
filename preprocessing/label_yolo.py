import pandas as pd
import os
from joblib import Parallel, delayed
import argparse
from tqdm import tqdm

def convert_bbox(xmin, ymin, xmax, ymax, w, h):
    cx = (xmin + xmax) / 2 / w
    cy = (ymin + ymax) / 2 / h
    bw = (xmax - xmin ) / w
    bh = (ymax- ymin) / h
    return cx, cy, bw ,bh

def get_classes():
    return {'Osteophytes':0, 'Spondylolysthesis':1, 'Disc space narrowing':2, 'Vertebral collapse':3, 'Foraminal stenosis':4,'Surgical implant':5, 'Other lesions':6}

def read_csv(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    return df

def process_img_csv(image_id, rows, meta, out_dir, classes_map):
    if image_id not in meta:
        return

    W = meta[image_id]['image_width']
    H = meta[image_id]['image_height']

    yolo_lines = []

    for _, r in rows.iterrows():
        if pd.isna(r.xmin):
            continue

        cls = classes_map.get(r.lesion_type, None)

        if cls is None:
            continue

        cx, cy, w, h = convert_bbox(
            r.xmin, r.ymin, r.xmax, r.ymax, W, H
        )

        if w <= 0 or h <=0:
            continue
        
        cx = min(max(cx, 0), 1)
        cy = min(max(cy, 0), 1)
        w = min(max(w, 0), 1)
        h = min(max(h, 0), 1)

        yolo_lines.append(
            f'{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}' 
        )

    if yolo_lines:
        with open(os.path.join(out_dir, f'{image_id}.txt'), 'w') as f:
            f.write('\n'.join(yolo_lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno-dir', required=True, type = str)
    parser.add_argument('--meta-dir', required=True, type = str)
    parser.add_argument('--output-dir', required=True, type = str)
    parser.add_argument('--cpus', default=2, type = int)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    ann_file = args.anno_dir
    meta_file = args.meta_dir
    out_dir = args.output_dir
    
    os.makedirs(out_dir, exist_ok= True)
    ann = read_csv(ann_file)
    meta_df = read_csv(meta_file)
    meta = (
    meta_df
    .drop_duplicates(subset='image_id')
    .set_index('image_id')
    .to_dict('index')
    )

    groups = list(ann.groupby('image_id'))
    cls = get_classes()

    if args.debug:
        groups = groups[:50]

    implmt = Parallel(
        n_jobs = args.cpus,
        backend='multiprocessing',
        prefer = 'processes',
        verbose = 1
    )

    do = delayed(process_img_csv)
    tasks = (do(image_id, rows, meta, out_dir, cls) for image_id, rows in tqdm(groups))
    implmt(tasks)


if __name__ == '__main__':
    main()