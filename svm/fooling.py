import numpy as np
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--pred_file", type=str)
parser.add_argument("--type", type=str, choices=("multi", "binary"))
args = parser.parse_args()

test_data = pd.read_csv("../data/testing.csv")
label, attack_cat = test_data['label'], test_data['attack_cat']
file = open(args.pred_file, "r")
pred = file.readlines()
pred = list(map(int, pred))
y, pred = np.array(label), np.array(pred)

diff = np.where(pred != label)[0]
print("acc:", 100 - len(diff) / len(pred) * 100)
filename = args.pred_file.split("/")[-1][:-4]
fooling = [attack_cat[idx] for idx in diff]

df = pd.DataFrame(
    {
        'id': diff,
        'attack_cat': fooling # 故意讓attack_cat在第二個
    }
)

if args.type == "multi":
    one_hot = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
    df['mis'] = [one_hot[pred[idx]] for idx in diff]

for key in test_data.keys():
    if key != "id" and key != "label" and key != 'attack_cat':
        df[key] = [test_data[key][idx] for idx in diff]

df.to_csv(args.type + "_" + filename + ".csv", index=False)
