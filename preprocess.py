import pandas as pd
import numpy as np

def normalize(X: np.ndarray) -> np.ndarray:
    X = (X - min(X)) / (max(X) - min(X))
    for i in range(len(X)):
        if np.isnan(X[i]):
            X[i] = 0
    return np.array(X)

no_normalize = ['id', 'proto', 'service', 'attack_cat', 'state', 'label']

def make_X(df) -> np.ndarray:
    x = [[] for i in range(len(df))]
    for key in df.keys():
        # print(key)
        if key != 'id' and key != 'label':
            for i, val in enumerate(df[key]):
                x[i].append(val)
    return np.array(x)

def load_data(file):
    data = pd.read_csv(file)
    for key in data.keys():
        if key not in no_normalize:
            data[key] = normalize(data[key])
    ## proto one hot
    proto = list(set(data['proto']))
    data['proto'] = [proto.index(i) for i in data['proto']]
    ## service one hot
    service = list(set(data['service']))
    data['service'] = [service.index(i) for i in data['service']]
    ## attack one hot
    attack = list(set(data['attack_cat']))
    print(attack)

    data['attack_cat'] = [attack.index(i) for i in data['attack_cat']]
    ## state one hot
    state = list(set(data['state']))
    data['state'] = [state.index(i) for i in data['state']]

    data = make_X(data)
    return data

def main():
    ## load train data
    train = load_data("./data/UNSW_NB15_training-set.csv")
    train_file = open("train", "w")
    for i, data in enumerate(train):
        assert(len(data) == 43)
        train_file.write(f"{data[-1]}") # attack
        for j in range(len(data)-1):
            train_file.write(f" {j+1}:{data[j]}")
        train_file.write("\n")
    train_file.close()
    print("training data loaded")
    ## load test data
    test = load_data("./data/UNSW_NB15_testing-set.csv")
    test_file = open("test", "w")
    for i, data in enumerate(test):
        assert(len(data) == 43)
        test_file.write(f"{data[-1]}") # attack
        for j in range(len(data)-1):
            test_file.write(f" {j+1}:{data[j]}")
        test_file.write("\n")
    test_file.close()
    print("testing data loaded")

if __name__ == "__main__":
    main()

