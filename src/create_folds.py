import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join('..', 'dataset', 'train.csv'))
    print(train_df.head())

    # Add a new column to store kfold indices
    train_df.loc[:, 'kfold'] = -1

    # Randomly shuffle the dataset
    train_df = train_df.sample(frac=1).reset_index(drop=True)

    X = train_df.image_id.values
    y = train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values

    mskf = MultilabelStratifiedKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
        train_df.loc[val_idx, 'kfold'] = fold
    
    print(train_df.kfold.value_counts())
    train_df.to_csv(os.path.join('..', 'dataset', 'train_folds.csv'), index=False)