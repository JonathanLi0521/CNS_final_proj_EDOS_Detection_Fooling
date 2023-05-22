import pandas as pd

def preprocessing(data, split):
    data = data.dropna()
    # min-max normalization
    no_normalize = ['id', 'proto', 'service', 'attack_cat', 'state', 'label']
    for column in data.columns:
        if column in no_normalize:
            continue
        data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())
    # one-hot encoding
    data = pd.get_dummies(data)
    data.to_csv(f'{split}.csv', index=False)

if __name__ == "__main__":
    train_data = pd.read_csv('../data/UNSW_NB15_training-set.csv')
    preprocessing(train_data, 'train')
    test_data = pd.read_csv("../data/UNSW_NB15_testing-set.csv")
    preprocessing(test_data, 'test')
