# Process UrbanSound8K

1. Set the sample rate of all the wav files to 20kHz.
2. All the clips set to 100k samples by padding and cropping clips.
3. Merge all the 10 classes into one folder and prepare 5 stratified folds and get a NPZ file as the resultant dataset.
4. Classes are 0-indexed.

## Instructions
1. Download the dataset from: https://www.kaggle.com/datasets/chrisfilo/urbansound8k
2. Create a folder named "urbansound8k" in the datasets directory and include the folders of 10 folds into that.
3. You can change the sample rate and the number of samples accordingly.
4. Create a python env, install the required libraries using:
   
   Run: ```pip install numpy librosa scikit-learn```.
5. Run the notebook "prepare_us8k.ipynb".
6. NPZ file will be saved in the datasets/urbansound8k directory.
