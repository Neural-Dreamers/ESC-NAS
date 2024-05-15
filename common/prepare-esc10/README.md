# Process ESC10

1. Set the sample rate of all the wav files to 20kHz.
2. All the clips set to 100k samples by padding and cropping clips.
3. Merge all the 10 classes into one folder and prepare 5 stratified folds and get a NPZ file as the resultant dataset.
4. Classes are 0-indexed
    * chainsaw: 0
    * clock_tick: 1
    * crackling_fire: 2
    * crying_baby: 3
    * dog: 4
    * helicopter: 5
    * rain: 6
    * rooster: 7
    * sea_waves: 8
    * sneezing: 9

## Instructions

1. Download the dataset from: https://www.kaggle.com/datasets/sreyareddy15/esc10rearranged
2. Create a folder named "esc10" in the datasets directory and extract the folders of 10 classes into that.
3. You can change the sample rate and the number of samples accordingly.
4. Run the notebook "prepare_esc10.ipynb".
5. NPZ file will be saved in the datasets/esc10 directory.

##### Note
* Downloading the datasets From Kaggle requires a [Kaggle](https://www.kaggle.com/) account.
Please create one if you already haven't.