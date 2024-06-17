# ESC-NAS
Environment Sound Classification with Deep Learning using Hardware-Aware - Neural Architecture Search 
for resource constrained edge devices.

### Published as:
ESC-NAS: Environment sound classification using hardware-aware Neural Architecture Search for the edge.
### Please cite this work as:
    @article{ranmal2024esc,
        title={ESC-NAS: Environment Sound Classification Using Hardware-Aware Neural Architecture Search for the Edge},
        author={Ranmal, Dakshina and Ranasinghe, Piumini and Paranayapa, Thivindu and Meedeniya, Dulani and Perera, Charith},
        journal={Sensors},
        volume={24},
        number={12},
        article-number={3749},
        url={https://www.mdpi.com/1424-8220/24/12/3749},
        pages={3749},
        year={2024},
        publisher={Multidisciplinary Digital Publishing Institute},
        doi={10.3390/s24123749}
    }

### Prerequisites
1. A valid Google Colab account to run the ESC-NAS process.
2. Create **python 3.11** development environment.
   `python -m venv env`
3. You can skip this step if you only run PyTorch on CPU.
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
4. You can skip this step if you run PyTorch on GPU (Executed the previous step).
   Install PyTorch on CPU.
   `pip install torch`
5. Install `FFmpeg` for downsampling and upsampling audio recordings.
6. Install other necessary dependencies.
   `pip install wavio wget pydub librosa matplotlib seaborn soundfile pandas scikit-learn`

##### Note
* ESC-NAS is developed and tested in Windows 10 environment and Google Colab. The forthcoming sections assumes that the above libraries/software are now installed.

## A. Dataset preparation
1. Download/clone the repository.
2. Go to the root of ESC-NAS directory using the terminal.
3. To process the required dataset, follow the instructions in the README files in
   `common/prepare-esc10`, `common/prepare-fsc22_esc50` and `common/prepare-us8k` directories.
4. Prepare the validation data for a given dataset, run: ```python common/val_generator.py --dataset <dataset_name>```.

*All the required data of the datasets for processing `44.1kHz` and `20kHz` are now ready at `datasets/<specific-dataset>` directory.*

## B. Executing ESC-NAS
*There are pretrained models resulting for each dataset from ESC-NAS are provided inside `torch/trained_models` directory that can be used instead of 
training a new model. The model names are self-explanatory*.

However, to conduct a brand new ESC-NAS process follow the instructions in the README files in
`nas` directory.

## C. Training the best model for a dataset resulting from ESC-NAS
1. To train a model resulting from ESC-NAS, first change the model architecture in `torch/resources/models.py`
2. Then run this command: ```python torch/trainer.py --dataset <dataset_name>``` to train the model for the given dataset.
3. Follow the on-screen self-explanatory steps.

## D. Testing an ESC-NAS model
1. To test a trained model, run this command: ```python torch/tester.py --dataset <dataset_name>``` to test the model for the given dataset.
2. Follow the on-screen self-explanatory steps.

##### Notes
* A model should always be tested on the fold on which it was validated to reproduce the result.
* For example, if a model was validated on fold-1, it will reproduce the validation accuracy on that fold.
For all other folds (fold 2-5), it will produce approximately 100% prediction accuracy since it was trained on those folds.
