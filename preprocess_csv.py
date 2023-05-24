from pathlib import Path
import json
import pandas as pd

preprocess_dir = Path('./preprocess/')
preprocess_dir.mkdir(parents=True, exist_ok=True)
dataset = {split: pd.read_csv(f'./data/UNSW_NB15_{split}ing-set.csv') for split in ['train', 'test']}
data = pd.concat([dataset['train'], dataset['test']], ignore_index=True)
data = data.dropna()

# attack_cat to multiclass label
attack2idx_path = preprocess_dir / 'attack2idx.json'
attack2idx = json.loads(attack2idx_path.read_text())
data['attack_cat'] = [attack2idx[attack_cat] for attack_cat in data['attack_cat']]

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

data[:dataset['train'].shape[0]].to_csv(preprocess_dir / 'train.csv', index=False)
data[dataset['train'].shape[0]:].to_csv(preprocess_dir / 'test.csv', index=False)
