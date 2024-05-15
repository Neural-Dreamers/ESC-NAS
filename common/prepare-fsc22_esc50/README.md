# Process FSC22 and ESC50

1. Rename all the audio clips to the format of - <fold_number>-<unique_id>-<label>.wav
2. Set the sample rate of all the wav files to 20kHz or 44kHz.
3. All the clips set to a nominal length samples by padding and cropping clips.
4. Create an NPZ file as the resultant dataset with fold split.
5. Classes are 0-indexed.

## Instructions

1. Set the correct URL of the dataset to be downloaded.
   * Get FSC22 from: https://storage.googleapis.com/kaggle-data-sets/2483929/4213460/bundle/archive.zip
   * Get ESC50 from: https://github.com/karolpiczak/ESC-50 
2. Change the code as required by the dataset and its directory structure.
   Set the correct URL of the datasets, directory names, Zip file names and directory paths.  
3. Run ```python common/prepare-fsc22_esc50/prepare_fsc22_esc50.py```
4. NPZ file will be saved in the datasets/<dataset_name> directory.

##### Note
* Downloading the datasets From Kaggle requires a [Kaggle](https://www.kaggle.com/) account.
Please create one if you already haven't.
