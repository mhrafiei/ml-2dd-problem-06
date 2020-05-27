```diff
! READ FIRST! 
```

# A. KERAS Package: open and run ml-2dd-problem-06/code-colab/code_colab_keras.ipynb in google colab

## 1. Clone the repository
>* !git config --global --unset http.proxy
>* !git config --global --unset https.proxy
>* !git config --global user.email "mrafiei1@jhu.edu"
>* !git config --global user.name "mrafiei1"
>* !git clone https://github.com/mhrafiei/ml-2dd-problem-06.git

## 2. Get the raw and filtered .mat and .txt data
>* cd /content/ml-2dd-problem-06/data-matlab/
>* !wget -O data_inou.mat https://www.dropbox.com/s/l7rtsxu9gmh6r0t/data_inou.mat?dl=0
>* !wget -O data_raw.mat https://www.dropbox.com/s/rfth5mzgxdhfqw9/data_raw.mat?dl=0

## 3. Download npy version of data and indices necessary for ML (prefered) or run /content/ml-2dd-problem-06/code-keras/code_data.py after adjusting rtt and rrs in it
>* cd /content/ml-2dd-problem-06/data-python/
>* !wget -O data.npy https://www.dropbox.com/s/9ez84xlwhgxrtsm/data.npy?dl=0
>* !wget -O ind.npy https://www.dropbox.com/s/pgr7cma27uupjlj/ind.npy?dl=0 
>* #!python3.6 /content/ml-2dd-problem-06/code-keras/code_data.py

## 4. Edit neuron architectures to be investigated in /content/ml-2dd-problem-06/code-keras/code_creator_keras.py and run it
>* cd /content/ml-2dd-problem-06/code-keras/
>* !python3.6 code_creator_keras.py


## 5. Edit ML input parameters in /content/ml-2dd-problem-06/code-keras/cls_keras.py and run any prefered /content/ml-2dd-problem-06/code-keras/master_keras_* file
>* #Example
>* !python3.6 master_keras_1_1_500_400_300.py
 
## 6. Results including the models and figures are availble at /content/ml-2dd-problem-06/data-python/
>* cd /content/ml-2dd-problem-06/data-python/

## P.S. Never delete temp.log files in subdirectories 


# B. SCIKIT Package: open and run ml-2dd-problem-06/code-colab/code_colab_scikit.ipynb in google colab


## 1. Clone the repository
>* !git config --global --unset http.proxy
>* !git config --global --unset https.proxy
>* !git config --global user.email "mrafiei1@jhu.edu"
>* !git config --global user.name "mrafiei1"
>* !git clone https://github.com/mhrafiei/ml-2dd-problem-06.git

## 2. Get the MATLAB raw data (data_raw.mat) and filtered data (data_inou.mat)
>* cd /content/ml-2dd-problem-06/data-matlab/
>* !wget -O data_inou.mat https://www.dropbox.com/s/l7rtsxu9gmh6r0t/data_inou.mat?dl=0
>* !wget -O data_raw.mat https://www.dropbox.com/s/rfth5mzgxdhfqw9/data_raw.mat?dl=0

## 3. Download npy version of data and indices necessary for ML (prefered) or run /content/ml-2dd-problem-06/code-scikit/code_data.py after adjusting rtt and rrs in it
>* cd /content/ml-2dd-problem-06/data-python/
>* !wget -O data.npy https://www.dropbox.com/s/9ez84xlwhgxrtsm/data.npy?dl=0
>* !wget -O ind.npy https://www.dropbox.com/s/pgr7cma27uupjlj/ind.npy?dl=0 
>* #!python3.6 /content/ml-2dd-problem-06/code-scikit/code_data.py

## 4. Edit neuron architectures to be investigated in /content/ml-2dd-problem-06/code-scikit/code_creator_scikit.py and run it
>* cd /content/ml-2dd-problem-06/code-scikit/
>* !python3.6 code_creator_scikit.py

## 5. Edit ML input parameters in /content/ml-2dd-problem-06/code-scikit/cls_scikit.py and run any prefered /content/ml-2dd-problem-06/code-scikit/master_scikit_* file
>* #Example
>* !python3.6 master_scikit_1_1_500_400_300.py
 
## 6. Results including the models and figures are availble at /content/ml-2dd-problem-06/data-python/
>* cd /content/ml-2dd-problem-06/data-python/

## P.S. Never delete temp.log files in subdirectories 


# C. KERAS-CNN Package: open and run ml-2dd-problem-06/code-colab/code_colab_keras_cnn.ipynb in google colab

## 1. Clone the repository
>* !git config --global --unset http.proxy
>* !git config --global --unset https.proxy
>* !git config --global user.email "mrafiei1@jhu.edu"
>* !git config --global user.name "mrafiei1"
>* !git clone https://github.com/mhrafiei/ml-2dd-problem-06.git

## 2. Download npy version of data and indices necessary for ML (prefered) or run /content/ml-2dd-problem-06/code-keras-cnn/code_data.py after adjusting rtt and rrs in it

>* cd /content/ml-2dd-problem-06/data-python/
>* !cat data_part_a* > data.zip
>* !unzip data.zip

## 3. Cat /content/ml-2dd-problem-06/data-python/data_part_* to /content/ml-2dd-problem-06/data-python/data.zip and unzip it
>* cd /content/ml-2dd-problem-06/code-keras-cnn/
>* !python3.6 code_creator_keras_cnn.py


## 4. Edit ML input parameters in /content/ml-2dd-problem-06/code-keras-cnn/cls_keras_cnn.py and run any prefered /content/ml-2dd-problem-06/code-keras-cnn/master_keras_cnn_* file
>* #Example
>* !python3.6 master_keras_cnn_1_1_64_32_16.py
 
## 5. Results including the models and figures are availble at /content/ml-2dd-problem-06/data-python/
>* cd /content/ml-2dd-problem-06/data-python/

## P.S. Never delete temp.log files in subdirectories 
