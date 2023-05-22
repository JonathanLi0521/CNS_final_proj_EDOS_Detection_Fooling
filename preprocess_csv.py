from pathlib import Path
import json
import pandas as pd

def preprocess(data, output_path):
    data = data.dropna()
    # min-max normalization
    no_normalize = ['id', 'proto', 'service', 'attack_cat', 'state', 'label']
    for column in data.columns:
        if column in no_normalize:
            continue
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    # one-hot encoding
    cat_columns = ['proto', 'service', 'state']
    one_hot = pd.get_dummies(data[cat_columns])
    data = data.drop(cat_columns, axis=1)
    data = data.join(one_hot)
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_dir = Path('./preprocess/')
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    # attack_cat to multiclass label
    attack2idx_path = preprocess_dir / 'attack2idx.json'
    attack2idx = json.loads(attack2idx_path.read_text())
    for split in ['train', 'test']:
        data = pd.read_csv(f'./data/UNSW_NB15_{split}ing-set.csv')
        data['attack_cat'] = [attack2idx[attack_cat] for attack_cat in data['attack_cat']]
        preprocess(data, preprocess_dir / f'{split}.csv')
