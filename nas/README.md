# Execute ESC-NAS
* Run the ESC-NAS process for the given dataset.
* Generate models and evaluate them.
* Output the initial filter count (k) and repetitive cell count (c) for the best model for the dataset
found from ESC-NAS.
* Save the evaluated models as .h5 models and their training metrics.


## Instructions
1. Create a folder names ESCNAS in your Google Drive.
2. Upload all the files in this directory to the ESCNAS folder (Do not upload the nas directory. Only the files inside it.).
3. Upload the `datasets` directory to Google Drive to the ESCNAS folder.
4. Run the ESC-NAS notebook for the required dataset using Google Colab.
5. Follow on-screen self-explanatory steps.
6. The trained models will be saved at `results_<dataset_name>_<timestamp>` directory.

##### Note
* Mount your Google Drive to access the file system and the stm32tflm simulator.
* Change directory to the location of the ESCNAS folder.