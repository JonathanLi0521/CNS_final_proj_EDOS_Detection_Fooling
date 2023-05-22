# CNS_final_proj_EDOS_Detection_Fooling
## Preprocess
```shell
python preprocess.py
```
* `preprocess.py` will generate two file, `train` and `test`, which are the inputs of libSVM.
* Details
  * For the row written in float or int, I did normalization.
  * For the row written in words, such as attack categories, I applied one hot to represent each category.

```shell
python preprocess_csv.py
```
* `preprocess_csv.py` generates two csv files, `train.csv` and `test.csv`.
* Details
  * Drop rows containing NA (seems none)
  * Min-max normalization
  * One hot encoding

## Grid Search to Reproduce Results

## Analysis
The current implementation of `analyze.py` takes one or two .csv files in the format of UNSW_NB15 as input. 
* For one file, comment the comparison section. Running `analyze.py` outputs `analysis` that contains the average and standard deviation of each continous data category.
* For two files, running `analyze.py` outputs `analysis` that contains the aforementioned information for both files, and also the difference between the two (both by value and by value / std).

## Environment Issues
* Running preprocess.py requires the module ordered_set
    If running the following command
    ```shell
    pip install ordered_set
    ```
    throws the error
    ```shell
    AttributeError: module 'lib' has no attribute 'X509_V_FLAG_CB_ISSUER_CHECK'
    ```
    , simply edit the crypto.py file mentioned in the stacktrace and remove the offending line with #.
    Then, if 
    ```shell
    pip install ordered_set
    ```
    throws the error
    ```shell
    AttributeError: module 'lib' has no attribute 'OpenSSL_add_all_algorithms'
    ```
    , uninstall pip with
    ```shell
    sudo apt remove python3-pip
    ```
    and run 
    ```shell
    sudo python3 get-pip.py
    ```
    Reboot and run 
    ```shell
    pip install pyopenssl --upgrade
    ```
    .Then, you should be able to install ordered_set and thus run preprocess.py

* Before training / grid search, run
    ```shell
    apt-get update
    ```
    and
    ``` shell
    apt install libsvm-tools
    ```
