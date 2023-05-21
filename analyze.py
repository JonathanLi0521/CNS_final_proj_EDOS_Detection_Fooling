import pandas as pd
import numpy as np
from ordered_set import OrderedSet

INFILEPATH = "./data/UNSW_NB15_training-set.csv"
INFILEPATH2 = "./data/UNSW_NB15_testing-set.csv"
OUTFILEPATH = "./analysis"

not_conti = ['id', 'proto', 'service', 'attack_cat', 'state', 'label']


def get_all_keys(procdata):
    all_keys = []
    for key in procdata:
        all_keys.append(key)
    return all_keys

def proc_conti(X):
    avg = np.mean(X)
    std = np.std(X)
    return (avg, std)

def proc_not_conti(X):
    pass
 
def analysis(procdata):
    
    analysis_res = {} # {key: (avg, var)} for continuous data
    for key in procdata.keys():
        if key in not_conti:
            proc_not_conti(procdata[key])
        else: 
            analysis_res[str(key)] = proc_conti(procdata[key])
    return analysis_res

# returns the difference of the averages between the two datasets, and the difference w.r.t. standard deviation of first dataset
def compare_analysis(analysis1, analysis2, keylist): 
    comp_res = {}
    for key in keylist:
        if key in not_conti:
            pass
        else:
            first_avg = analysis1[str(key)][0]
            first_std = analysis1[str(key)][1]
            sec_avg = analysis2[str(key)][0]
            avgdiff = first_avg - sec_avg
            avgstddiff = avgdiff / first_std
            comp_res[str(key)] = (avgdiff, avgstddiff)
    return comp_res


def main():
    output = open(OUTFILEPATH, "w+")
    procdata1 = pd.read_csv(INFILEPATH)
    
    all_keys = get_all_keys(procdata1)
    analysis1 = analysis(procdata1)
    
    output.write(str(INFILEPATH) + ":\n")
    output.write(str(analysis1) + "\n\n")

    # if you want to compare two sets, uncomment this section
    procdata2 = pd.read_csv(INFILEPATH2)
    analysis2 = analysis(procdata2)

    output.write(str(INFILEPATH2) + ":\n")
    output.write(str(analysis2) + "\n\n")

    compare_res = compare_analysis(analysis1, analysis2, all_keys)
    output.write("comparison:\n")
    output.write(str(compare_res) + "\n\n")
    

if __name__ == "__main__":
    main()

