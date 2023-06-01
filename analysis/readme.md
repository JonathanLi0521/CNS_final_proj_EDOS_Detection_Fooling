# Analyzing Datasets and Their Differences
* The goal of this section is to obtain information on a target file's features' averages and standard deviations. The files should be in the same format as the training set of UNSW-NB15.
## Analyze.py
In addition, Analyze.py compares two files, outputting the difference between the average of each feature, and also calculates the average divided by the standard deviation of the first file.
```shell
python analyze.py --comp_file_1 <path to first input file> --comp_file_2 <path to second input file> --outputcsv <path to output file (csv)> --mode compare
```
## Analyze2.py
Analyze2.py compares three files, additionally outputting the sorted (argsort according to standard deviation of file two and file three) information of both the individual file analysis and the comparisons between file 1 - file 2 and file 1 - file3.
```shell
python3 analyze2.py --comp_file_1 <path to first input file> --comp_file_2 <path to second input file> --comp_file_3 <path to thrid input file> --outputcsv <path to output file(csv)>
```