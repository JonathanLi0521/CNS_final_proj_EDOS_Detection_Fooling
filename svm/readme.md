# Fooling Case of SVM
* All you have to do is to run the following command with different `pred_file`. Don't worried about any filename collision.
## Binary Classification
```shell
python fooling.py --pred_file binary/out_100_1.txt --type binary # You can try other output file in binary/
```
## Multiclass Classification
```shell
python fooling.py --pred_file multi/out_10000_100.txt --type multi # You can try other output file in multi/
```