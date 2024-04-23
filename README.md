# ESC-NAS
Environment Sound Classification with Deep Learning using Hardware-Aware - Neural Architecture Search 
for resource constrained edge devices.

### Published as:
ESC-NAS: Environment sound classification using hardware-aware Neural Architecture Search for the edge.
### Please cite this work as:
    @article{
    }

## A. Training and Compressing ESC-NAS

#### A.1 Prerequisits
(pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 wavio wget pydub librosa matplotlib seaborn soundfile pandas scikit-learn)
1. Create python 3.11 conda environment named tf.
   `conda create --name tf python=3.11`
2. Activate the environment.
   `conda activate tf`
3. You can skip this step if you only run TensorFlow on CPU.

   First install NVIDIA GPU https://www.nvidia.com/Download/index.aspx driver if you have not.
   Then install the CUDA, cuDNN with conda.
   `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
4. Install tensorflow `python -m pip install "tensorflow<2.11"`
5. Install `FFmpeg` for downsampling and upsampling audio recordings.
6. Install other necessary dependencies.
   `pip install wavio wget pydub librosa matplotlib seaborn soundfile pandas`

##### Note
* ESC-NAS is developed and tested in Windows 10 environment. The forthcoming sections assumes that the above libraries/software are now installed.*

#### A.2 Dataset preparation
1. Download/clone the repository.
2. Go to the root of ESC-NAS directory using the terminal.
3. To process the required dataset, follow the instructions in the README files in
   `common/prepare-esc10`, `common/prepare-fsc22_esc50` and `common/prepare-us8k` directories.
4. Prepare the validation data, run: ```python common/val_generator.py```.
5. Prepare the independent test data, run: ```python common/test_generator.py```.

*All the required data of FSC-22 for processing `44.1kHz` and `20kHz` are now ready at `datasets/fsc22` directory*

#### A.3 Training ESC-NAS
*There are pretrained models provided inside `tf/resources/pretrained_models` directory that can be used instead of 
training a new model. The model names are self-explanatory*.

However, to conduct the training of a brand new ESC-NAS, run: ```python tf/trainer.py```.
##### Notes
* Follow on-screen self-explanatory steps.
* To train a brand new ESC-NAS, please select `training from scratch` option and keep the model path `empty` in the next step.
* The trained models will be saved at `tf/trained_models directory`.
* The models will have names `YourGivenName_foldNo` on which it was validated.
* For five-fold cross validation, there will be 5 models named accordingly.

#### A.4 Testing ESC-NAS
1. To test a trained model, run this command: ```python tf/tester.py```.
2. Follow the on-screen self-explanatory steps.

##### Notes
* A model should always be tested on the fold on which it was validated to reproduce the result.
* For example, if a model was validated on fold-1, it will reproduce the validation accuracy on that fold.
For all other folds (fold 2-5), it will produce approximately 100% prediction accuracy since it was trained on those folds.