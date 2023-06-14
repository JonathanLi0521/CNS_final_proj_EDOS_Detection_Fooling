import numpy as np
import pandas as pd
import csv
import math
from argparse import ArgumentParser
BIGPOSNUM = 999999999999999999
NEGNUM = -99999999999999

parser = ArgumentParser()
parser.add_argument("--comp_file_1", type=str)
parser.add_argument("--comp_file_2", type=str)
parser.add_argument("--comp_file_3", type=str)
parser.add_argument("--outputcsv", type=str)
# parser.add_argument("--type", type=str, choices=("multi", "binary"))
args = parser.parse_args()

not_conti = ['id', 'proto', 'service', 'attack_cat', 'state', 'label', 'mis']

comp1 = pd.read_csv(args.comp_file_1) # usually the normal case
comp2 = pd.read_csv(args.comp_file_2) # 
comp3 = pd.read_csv(args.comp_file_3)

def proc_conti(X):
    avg = np.mean(X)
    std = np.std(X)
    return (avg, std)

def proc_not_conti(X):
    return (NEGNUM, NEGNUM)

def metric_calculation(X):
    negcount = 0
    poscount = 0
    squaresum = 0
    max = 0
    for item in X:
        if item == NEGNUM: negcount += 1
        elif item == BIGPOSNUM: poscount += 1 
        else: 
            squaresum += item ** 2
            if item ** 2 > max: max = item ** 2
    # squaresum += max * poscount
    ans = math.sqrt(squaresum) / (len(X) - negcount - poscount)
    print("negcount: {}, poscount: {}".format(negcount, poscount))
    return ans


    

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
    
    print(args.outputcsv)
    print("normal " + str(metric_calculation(stddiffrow)))

    avgrow3 = []
    stdrow3 = []
    for column in comp3.columns:
        if column not in not_conti:
            avg, std = proc_conti(np.array(comp3[column]))
        else:
            avg, std = proc_not_conti(np.array(comp3[column]))
        avgrow3.append(avg)
        stdrow3.append(std)
    avgdiffrow3 = []
    stddiffrow3 = []
    for i in range(len(avgrow1)):
        if(avgrow1[i] == NEGNUM):
            avgdiffrow3.append(NEGNUM)
            stddiffrow3.append(NEGNUM)
        else:
            avgdiffrow3.append(float(avgrow3[i]) - float(avgrow1[i]))
            if (float(stdrow1[i]) == 0):
                if (avgdiffrow3[i] == 0):
                    stddiffrow3.append(0)
                else:
                    stddiffrow3.append(BIGPOSNUM)
            else:
                stddiffrow3.append(avgdiffrow3[i] / float(stdrow1[i]))
    print("fooling " + str(metric_calculation(stddiffrow3)))
    print()
    # very dirty code that deals with using 2nd file to sort
    absolutestddiffrow3 = np.absolute(stddiffrow3)
    for i in range(len(absolutestddiffrow3)):
        if absolutestddiffrow3[i] == -NEGNUM:
            absolutestddiffrow3[i] = NEGNUM
    sort3 = np.argsort(absolutestddiffrow3)
    sort3 = sort3[::-1]
    Aavgrow1 = list(np.array(avgrow1)[sort3])
    Aavgrow2 = list(np.array(avgrow2)[sort3])
    Astdrow1 = list(np.array(stdrow1)[sort3])
    Astdrow2 = list(np.array(stdrow2)[sort3])
    Aavgdiffrow = list(np.array(avgdiffrow)[sort3])
    Astddiffrow = list(np.array(stddiffrow)[sort3])
    Aavgrow3 = list(np.array(avgrow3)[sort3])
    Astdrow3 = list(np.array(stdrow3)[sort3])
    Aavgdiffrow3 = list(np.array(avgdiffrow3)[sort3])
    Astddiffrow3 = list(np.array(stddiffrow3)[sort3])
    Acompare_categories = list(np.array(compare_categories)[sort3])
    
    # using first file to sort
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
    avgrow3 = list(np.array(avgrow3)[sort])
    stdrow3 = list(np.array(stdrow3)[sort])
    avgdiffrow3 = list(np.array(avgdiffrow3)[sort])
    stddiffrow3 = list(np.array(stddiffrow3)[sort])
    compare_categories = list(np.array(compare_categories)[sort])
    
    # file writing
    csv_writer = csv.writer(outfile)
    csv_writer.writerow(["sorted by 1st:"])
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
        if (avgrow3[i] == NEGNUM):
            avgrow3[i] = "X"
        if (avgrow3[i] == NEGNUM):
            avgrow3[i] = "X"
        if (avgdiffrow3[i] == NEGNUM):
            avgdiffrow3[i] = "X"
        if (stddiffrow3[i] == NEGNUM):    
            stddiffrow3[i] = "X"
        if (Aavgrow1[i] == NEGNUM):
            Aavgrow1[i] = "X"
        if (Astdrow1[i] == NEGNUM):    
            Astdrow1[i] = "X"
        if (Aavgrow2[i] == NEGNUM):
            Aavgrow2[i] = "X"
        if (Astdrow2[i] == NEGNUM):    
            Astdrow2[i] = "X"
        if (Aavgdiffrow[i] == NEGNUM):
            Aavgdiffrow[i] = "X"
        if (Astddiffrow[i] == NEGNUM):    
            Astddiffrow[i] = "X"
        if (Aavgrow3[i] == NEGNUM):
            Aavgrow3[i] = "X"
        if (Aavgrow3[i] == NEGNUM):
            Aavgrow3[i] = "X"
        if (Aavgdiffrow3[i] == NEGNUM):
            Aavgdiffrow3[i] = "X"
        if (Astddiffrow3[i] == NEGNUM):    
            Astddiffrow3[i] = "X"
    
    avgrow1.insert(0, "avg")
    stdrow1.insert(0, "std")
    csv_writer.writerow(avgrow1)
    csv_writer.writerow(stdrow1)

    
    csv_writer.writerow([])
    csv_writer.writerow([args.comp_file_2])
    csv_writer.writerow(compare_categories)
    avgrow2.insert(0, "avg")
    stdrow2.insert(0, "std")
    csv_writer.writerow(avgrow2)
    csv_writer.writerow(stdrow2)

    csv_writer.writerow([])
    csv_writer.writerow([args.comp_file_3])
    csv_writer.writerow(compare_categories)
    avgrow3.insert(0, "avg")
    stdrow3.insert(0, "std")
    csv_writer.writerow(avgrow3)
    csv_writer.writerow(stdrow3)
    
    
    csv_writer.writerow([])
    csv_writer.writerow(["compare 1st with normal"])
    csv_writer.writerow(compare_categories)
    avgdiffrow.insert(0, "avgdiff")
    stddiffrow.insert(0, "stddiff")
    csv_writer.writerow(avgdiffrow)
    csv_writer.writerow(stddiffrow)

    csv_writer.writerow([])
    csv_writer.writerow(["compare 2nd with normal"])
    csv_writer.writerow(compare_categories)
    avgdiffrow3.insert(0, "avgdiff")
    stddiffrow3.insert(0, "stddiff")
    csv_writer.writerow(avgdiffrow3)
    csv_writer.writerow(stddiffrow3)



    csv_writer.writerow([])
    csv_writer.writerow([])
    csv_writer.writerow(["sorted by 2nd:"])
    csv_writer.writerow([args.comp_file_1])
    Acompare_categories.insert(0, "")
    csv_writer.writerow(Acompare_categories)
    Aavgrow1.insert(0, "avg")
    Astdrow1.insert(0, "std")
    csv_writer.writerow(Aavgrow1)
    csv_writer.writerow(Astdrow1)

    csv_writer.writerow([])
    csv_writer.writerow([args.comp_file_2])
    csv_writer.writerow(Acompare_categories)
    Aavgrow2.insert(0, "avg")
    Astdrow2.insert(0, "std")
    csv_writer.writerow(Aavgrow2)
    csv_writer.writerow(Astdrow2)

    csv_writer.writerow([])
    csv_writer.writerow([args.comp_file_3])
    csv_writer.writerow(Acompare_categories)
    Aavgrow3.insert(0, "avg")
    Astdrow3.insert(0, "std")
    csv_writer.writerow(Aavgrow3)
    csv_writer.writerow(Astdrow3)
    
    
    csv_writer.writerow([])
    csv_writer.writerow(["compare 1st with normal"])
    csv_writer.writerow(Acompare_categories)
    Aavgdiffrow.insert(0, "avgdiff")
    Astddiffrow.insert(0, "stddiff")
    csv_writer.writerow(Aavgdiffrow)
    csv_writer.writerow(Astddiffrow)

    csv_writer.writerow([])
    csv_writer.writerow(["compare 2nd with normal"])
    csv_writer.writerow(Acompare_categories)
    Aavgdiffrow3.insert(0, "avgdiff")
    Astddiffrow3.insert(0, "stddiff")
    csv_writer.writerow(Aavgdiffrow3)
    csv_writer.writerow(Astddiffrow3)

# test_data = pd.read_csv("../data/testing.csv")
