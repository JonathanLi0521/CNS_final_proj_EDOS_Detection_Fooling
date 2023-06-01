import numpy as np
import pandas as pd
import csv
from argparse import ArgumentParser
BIGPOSNUM = 999999999999999999
NEGNUM = -99999999999999

parser = ArgumentParser()
parser.add_argument("--comp_file_1", type=str)
parser.add_argument("--comp_file_2", type=str)
parser.add_argument("--outputcsv", type=str)
parser.add_argument("--mode", type=str)
# parser.add_argument("--type", type=str, choices=("multi", "binary"))
args = parser.parse_args()

not_conti = ['id', 'proto', 'service', 'attack_cat', 'state', 'label', 'mis']

comp1 = pd.read_csv(args.comp_file_1)

if(args.mode == 'compare'):
    comp2 = pd.read_csv(args.comp_file_2)

def proc_conti(X):
    avg = np.mean(X)
    std = np.std(X)
    return (avg, std)

def proc_not_conti(X):
    return (NEGNUM, NEGNUM)

compare_categories = list(comp1.columns)


with open(args.outputcsv, 'w+') as outfile:
    avgrow1 = []
    stdrow1 = []
    for column in comp1.columns:
        if column not in not_conti:
            avg, std = proc_conti(np.array(comp1[column]))
        else:
            avg, std = proc_not_conti(np.array(comp1[column]))
        avgrow1.append(avg)
        stdrow1.append(std)
    
    if(args.mode == 'compare'):
        avgrow2 = []
        stdrow2 = []
        for column in comp2.columns:
            if column not in not_conti:
                avg, std = proc_conti(np.array(comp2[column]))
            else:
                avg, std = proc_not_conti(np.array(comp2[column]))
            avgrow2.append(avg)
            stdrow2.append(std)
        avgdiffrow = []
        stddiffrow = []
        for i in range(len(avgrow1)):
            if(avgrow1[i] == NEGNUM):
                avgdiffrow.append(NEGNUM)
                stddiffrow.append(NEGNUM)
            else:
                avgdiffrow.append(float(avgrow2[i]) - float(avgrow1[i]))
                if (float(stdrow1[i]) == 0):
                    if (avgdiffrow[i] == 0):
                        stddiffrow.append(0)
                    else:
                        stddiffrow.append(BIGPOSNUM)
                else:
                    stddiffrow.append(avgdiffrow[i] / float(stdrow1[i]))
    
    absolutestddiffrow = np.absolute(stddiffrow)
    for i in range(len(absolutestddiffrow)):
        if absolutestddiffrow[i] == -NEGNUM:
            absolutestddiffrow[i] = NEGNUM
    sort = np.argsort(absolutestddiffrow)
    sort = sort[::-1]
    avgrow1 = list(np.array(avgrow1)[sort])
    avgrow2 = list(np.array(avgrow2)[sort])
    stdrow1 = list(np.array(stdrow1)[sort])
    stdrow2 = list(np.array(stdrow2)[sort])
    avgdiffrow = list(np.array(avgdiffrow)[sort])
    stddiffrow = list(np.array(stddiffrow)[sort])
    compare_categories = list(np.array(compare_categories)[sort])
    
    # file writing
    csv_writer = csv.writer(outfile)
    compare_categories.insert(0, '')
    csv_writer.writerow([args.comp_file_1])
    csv_writer.writerow(compare_categories)
    
    for i in range(len(avgrow1)):
        if (avgrow1[i] == NEGNUM):
            avgrow1[i] = "X"
        if (stdrow1[i] == NEGNUM):    
            stdrow1[i] = "X"
        if (avgrow2[i] == NEGNUM):
            avgrow2[i] = "X"
        if (stdrow2[i] == NEGNUM):    
            stdrow2[i] = "X"
        if (avgdiffrow[i] == NEGNUM):
            avgdiffrow[i] = "X"
        if (stddiffrow[i] == NEGNUM):    
            stddiffrow[i] = "X"
    avgrow1.insert(0, "avg")
    stdrow1.insert(0, "std")
    csv_writer.writerow(avgrow1)
    csv_writer.writerow(stdrow1)
    if(args.mode == 'compare'):
        csv_writer.writerow([])
        csv_writer.writerow([args.comp_file_2])
        csv_writer.writerow(compare_categories)
        avgrow2.insert(0, "avg")
        stdrow2.insert(0, "std")
        csv_writer.writerow(avgrow2)
        csv_writer.writerow(stdrow2)
        csv_writer.writerow([])
        csv_writer.writerow(["compare"])
        csv_writer.writerow(compare_categories)
        avgdiffrow.insert(0, "avgdiff")
        stddiffrow.insert(0, "stddiff")
        csv_writer.writerow(avgdiffrow)
        csv_writer.writerow(stddiffrow)

# test_data = pd.read_csv("../data/testing.csv")
