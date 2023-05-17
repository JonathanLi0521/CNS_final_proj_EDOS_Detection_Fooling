# CNS_final_proj_EDOS_Detection_Fooling
## Preprocess
```shell
python preprocess.py
```
* `preprocess.py` will generate two file, `train` and `test`, which are the inputs of libSVM.
* Details
  * For the row written in float or int, I did normalization.
  * For the row written in words, such as attack categories, I applied one hot to represent each category.
