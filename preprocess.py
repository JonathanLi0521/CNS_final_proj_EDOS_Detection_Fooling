import pandas as pd
import numpy as np
from ordered_set import OrderedSet

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

def write_file(filename, df):
    file = open(filename, "w")
    for i, data in enumerate(df):
        assert(len(data) == 43)
        file.write(f"{data[-1]}") # attack
        for j in range(len(data)-1):
            file.write(f" {j+1}:{data[j]}")
        file.write("\n")
    file.close()

def load_data():
    ## training train_data
    train_data = pd.read_csv("./data/UNSW_NB15_training-set.csv")
    for key in train_data.keys():
        if key not in no_normalize:
            train_data[key] = normalize(train_data[key])
    print("training normalization complete")
    ## testing data
    test_data = pd.read_csv("./data/UNSW_NB15_testing-set.csv")
    for key in test_data.keys():
        if key not in no_normalize:
            test_data[key] = normalize(test_data[key])
    print("testing normalization complete")
    ## proto one hot
    proto = list(OrderedSet(train_data['proto']).union(OrderedSet(test_data['proto'])))
    train_data['proto'] = [proto.index(i) for i in train_data['proto']]
    test_data['proto'] = [proto.index(i) for i in test_data['proto']]
    ## service one hot
    service = list(OrderedSet(train_data['service']).union(OrderedSet(test_data['service'])))
    train_data['service'] = [service.index(i) for i in train_data['service']]
    test_data['service'] = [service.index(i) for i in test_data['service']]
    ## attack one hot
    attack = list(OrderedSet(train_data['attack_cat']).union(OrderedSet(test_data['attack_cat'])))
    train_data['attack_cat'] = [attack.index(i) for i in train_data['attack_cat']]
    test_data['attack_cat'] = [attack.index(i) for i in test_data['attack_cat']]
    ## state one hot
    state = list(OrderedSet(train_data['state']).union(OrderedSet(test_data['state'])))
    train_data['state'] = [state.index(i) for i in train_data['state']]
    test_data['state'] = [state.index(i) for i in test_data['state']]
    print("one hot complete")
    ## write to training file
    train_data = make_X(train_data)
    write_file("train", train_data)
    print("write to the training file")    
    ## write to testing file
    test_data = make_X(test_data)
    write_file("test", test_data)
    print("write to the testing file")

def main():
    load_data()

if __name__ == "__main__":
    main()

