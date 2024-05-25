"""
 Usages: Dataset preparation code for Dataset - Ex: FSC-22
 Change the code accordingly to prepare the other datasets
 Prerequisites: FFmpeg and wget need to be installed.
"""

import glob
import os
import shutil
import zipfile

import numpy as np
import pydub
import wavio
import wget


def main():
    main_dir = os.getcwd()
    dataset_path = os.path.join(main_dir, 'datasets/fsc22')  # change the dataset name accordingly

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    sr_list = [44100, 20000]

    # Set the URL of the dataset. Change this accordingly
    url = 'https://storage.googleapis.com/kaggle-data-sets/2483929/4213460/bundle/archive.zip'

    # Set the save location for the dataset
    save_location = dataset_path

    # Download the dataset.
    wget.download(url, save_location)

    # Unzip the dataset. Change the name of the zip file accordingly.
    zip_file = "archive.zip"
    with zipfile.ZipFile(dataset_path + "\\" + zip_file, "r") as zip_ref:
        zip_ref.extractall(save_location)

    # Remove the zip file
    # os.remove(dataset_path + "\\" + zip_file)

    # Change the master file directory name accordingly
    dataset_master_path = os.path.join(dataset_path, 'FSC-22-master')

    # Comment out the lines from 48 to 49 when preparing ESC-50
    if not os.path.exists(dataset_master_path):
        shutil.copytree(os.path.join(dataset_path, 'Audio Wise V1.0-20220916T202003Z-001'), dataset_master_path)

    dataset_master_audio_path = os.path.join(dataset_master_path, 'audio')

    # Comment out the lines from 54 to 58 when preparing ESC-50
    if not os.path.exists(dataset_master_audio_path):
        os.rename(os.path.join(dataset_master_path, 'Audio Wise V1.0'), os.path.join(dataset_master_path, 'audio'))

        # rename audio files and split into folds
        rename_source_files(dataset_master_audio_path)

    # Convert sampling rate
    for sr in sr_list:
        convert_sr(os.path.join(dataset_master_audio_path),
                   os.path.join(dataset_path, 'wav{}'.format(sr // 1000)),
                   sr)

    # Create npz files
    for sr in sr_list:
        src_path = os.path.join(dataset_path, 'wav{}'.format(sr // 1000))

        create_dataset(src_path, os.path.join(dataset_path, 'wav{}.npz'.format(sr // 1000)))


def rename_source_files(src_path):
    folds = 5
    audio_file_list = sorted(os.listdir(src_path))

    for fold in range(1, folds + 1):
        for i in range(fold - 1, len(audio_file_list), folds):
            audio_file = audio_file_list[i]
            label = audio_file.split('_')[0]

            # skip the audio files with label 16 - wood chop due to ambiguities - Only for FSC-22
            if label == '16':
                continue
            # decrement the labels that are greater than 16 - due to the removal of label 16 - Only for FSC-22
            if int(label) > 16:
                label = str(int(label) - 1)

            index = audio_file.split('_')[1].split('.')[0]
            new_filename = str(fold) + '-' + index + '-' + label + '.wav'
            os.rename(os.path.join(src_path, audio_file), os.path.join(src_path, new_filename))

    for filename in glob.glob(os.path.join(src_path, "*_*")):
        os.remove(filename)


def convert_sr(src_path, dst_path, sr):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path)
        # Create an AudioSegment object
        audio_segment = pydub.AudioSegment.from_file(src_file)

        # Set the audio channels to 1
        audio_segment = audio_segment.set_channels(1)

        # Set the audio sample rate to sr
        audio_segment = audio_segment.set_frame_rate(sr)

        # Export the file to dst_file
        audio_segment.export(dst_file, format="wav")


def create_dataset(src_path, dataset_dst_path):
    print('* {} -> {}'.format(src_path, dataset_dst_path))
    dataset = {}

    for fold in range(1, 6):
        dataset['fold{}'.format(fold)] = {}
        sounds = []
        labels = []

        for wav_file in sorted(glob.glob(os.path.join(src_path, '{}-*.wav'.format(fold)))):
            sound = wavio.read(wav_file).data.T[0]
            label = int(os.path.splitext(wav_file)[0].split('-')[-1])
            sounds.append(sound)
            labels.append(label)

        dataset['fold{}'.format(fold)]['sounds'] = sounds
        dataset['fold{}'.format(fold)]['labels'] = labels

    all_labels = {label for fold_data in dataset.values() for label in fold_data['labels']}
    print(f'before: {all_labels}')

    # make labels 0 indexed
    if min(all_labels) == 1:
        print('Dataset is 1 indexed')
        for fold_data in dataset.values():
            fold_data['labels'] = [label - 1 for label in fold_data['labels']]

    all_labels = {label for fold_data in dataset.values() for label in fold_data['labels']}
    print(f'after: {all_labels}')

    np.savez(dataset_dst_path, **dataset)


if __name__ == '__main__':
    main()
