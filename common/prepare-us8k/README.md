# Process UrbanSound8K

1. Set the sample rate of all the wav files to 20kHz.
2. All the clips set to 100k samples by padding and cropping clips.
3. Create an NPZ file as the resultant dataset with fold split.
4. Classes are 0-indexed.

## Instructions
1. Download the dataset from: https://www.kaggle.com/datasets/chrisfilo/urbansound8k
2. Create a folder named "urbansound8k" in the datasets directory and extract the downloaded zip file into that.
3. You can change the sample rate and the number of samples accordingly.
4. Run the notebook "prepare_us8k.ipynb".
5. NPZ file will be saved in the datasets/urbansound8k directory.


##### Note
* Downloading the datasets From Kaggle requires a [Kaggle](https://www.kaggle.com/) account.
Please create one if you already haven't.