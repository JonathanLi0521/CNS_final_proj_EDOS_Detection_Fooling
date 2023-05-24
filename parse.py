import os

INFILEPATH = "./data/UNSW_NB15_testing-set.csv"
seen_categories = {}
pathlist = []

def gen_out_path(category):
    path = "./data/testing_" + category + ".csv"
    seen_categories[category] = len(pathlist)
    pathlist.append(path)
    
def gen_out_files():
    filelist = [open(f, 'a') for f in pathlist]
    return filelist
    
def main():
    infile = open(INFILEPATH, "r")
    flag = 0
    for line in infile:
        if flag == 0:
            flag = 1
            continue
        line = line.split(',') 
        if(str(line[-2]) not in seen_categories):# line[-2] is the attack category
            gen_out_path(str(line[-2]))
    infile.close()
    print(seen_categories)
    print(pathlist)
    filelist = gen_out_files()
    print(filelist)
    infile = open(INFILEPATH, "r")
    for line in infile:
        if flag == 1:
            flag = 2
            for f in filelist:
                f.write(str(line))
            continue
        linetmp = line.split(',')
        filelist[seen_categories[str(linetmp[-2])]].write(str(line))
    
if __name__ == "__main__":
    main()


