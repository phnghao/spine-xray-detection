import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def split_csv(annotation_file, out_dir,
              train_ratio=0.85,
              seed=36):

    assert 0 < train_ratio < 1.0
    val_ratio = 1.0 - train_ratio 

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(annotation_file)
    image_ids = df['image_id'].unique()

    # ---- 1. Split test ----
    train_ids, val_ids = train_test_split(
        image_ids,
        train_size = train_ratio,
        random_state=seed,
        shuffle=True
    )

    train_df = df[df.image_id.isin(train_ids)]
    val_df   = df[df.image_id.isin(val_ids)]

    train_df.to_csv(f'{out_dir}/train.csv', index=False)
    val_df.to_csv(f'{out_dir}/val.csv', index=False)

    print(f"Train images: {len(train_ids)}")
    print(f"Val images:   {len(val_ids)}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--anno-file', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--train-ratio', default=0.85, type=float)
    parser.add_argument('--seed', default=36, type=int)

    args = parser.parse_args()

    split_csv(
        annotation_file=args.anno_file,
        out_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main()