import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def split_csv(annotation_file, out_dir,
              train_ratio=0.7,
              val_ratio=0.15,
              test_ratio=0.15,
              seed=36):

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(annotation_file)
    image_ids = df['image_id'].unique()

    # ---- 1. Split test ----
    trainval_ids, test_ids = train_test_split(
        image_ids,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True
    )

    # ---- 2. Split train / val ----
    val_size = val_ratio / (train_ratio + val_ratio)

    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=val_size,
        random_state=seed,
        shuffle=True
    )

    train_df = df[df.image_id.isin(train_ids)]
    val_df   = df[df.image_id.isin(val_ids)]
    test_df  = df[df.image_id.isin(test_ids)]

    train_df.to_csv(f'{out_dir}/train.csv', index=False)
    val_df.to_csv(f'{out_dir}/val.csv', index=False)
    test_df.to_csv(f'{out_dir}/test.csv', index=False)

    print(f"Train images: {len(train_ids)}")
    print(f"Val images:   {len(val_ids)}")
    print(f"Test images:  {len(test_ids)}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--anno-file', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--train-ratio', default=0.7, type=float)
    parser.add_argument('--val-ratio', default=0.15, type=float)
    parser.add_argument('--test-ratio', default=0.15, type=float)
    parser.add_argument('--seed', default=36, type=int)

    args = parser.parse_args()

    split_csv(
        annotation_file=args.anno_file,
        out_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

if __name__ == '__main__':
    main()