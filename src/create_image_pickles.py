import os
import glob
import joblib

import pandas as pd
import numpy as np

from tqdm import tqdm

if __name__ == '__main__':
    files = glob.glob(os.path.join('..', 'dataset', 'train_*.parquet'))

    for f in files:
        train_df = pd.read_parquet(f)

        image_ids = train_df.image_id.values
        train_df = train_df.drop('image_id', axis=1)
        image_array = train_df.values
        for idx, img_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[idx, :], os.path.join('..', 'dataset', 'image_pickles', f'{img_id}.pkl'))
