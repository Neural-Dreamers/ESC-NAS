import os
import random
import sys
import time

import numpy as np

import opts
import utils as u


class ValGenerator:
    # Generates data for Keras
    def __init__(self, samples, labels, options):
        random.seed(42)
        # Initialization
        self.data = [(samples[i], labels[i]) for i in range(0, len(samples))]
        self.opt = options
        self.batch_size = options.batchSize // options.nCrops
        self.preprocess_funcs = self.preprocess_setup()

    def get_data(self):
        # Generate one batch of data
        x, y = self.generate()
        x = np.expand_dims(x, axis=1)
        x = np.expand_dims(x, axis=3)
        return x, y

    def generate(self):
        # Generates data containing batch_size samples
        sound_list = []
        label_list = []
        indexes = None
        for i in range(len(self.data)):
            sound, target = self.data[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.zeros((self.opt.nCrops, self.opt.nClasses[self.opt.dataset]))
            label[:, target-1] = 1

            sound_list.append(sound)
            label_list.append(label)

        sound_list = np.asarray(sound_list)
        label_list = np.asarray(label_list)

        sound_list = sound_list.reshape(sound_list.shape[0] * sound_list.shape[1], sound_list.shape[2])
        label_list = label_list.reshape(label_list.shape[0] * label_list.shape[1], label_list.shape[2])

        return sound_list, label_list

    def preprocess_setup(self):
        funcs = []
        funcs += [u.padding(self.opt.inputLength // 2),
                  u.normalize(32768.0),
                  u.multi_crop(self.opt.inputLength, self.opt.nCrops)]

        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound


if __name__ == '__main__':
    opt = opts.parse()
    opts.display_info(opt)
    opt.batchSize = opt.nSamples[opt.dataset]

    for sr in [44100, 20000]:
        if opt.dataset not in ['fsc22', 'esc50'] and sr == 44100:
            continue
        opt.sr = sr
        opt.inputLength = 66650 if sr == 44100 else 30225
        mainDir = os.getcwd()
        test_data_dir = os.path.join(mainDir, 'datasets/{}/test_data_{}khz'.format(opt.dataset, sr // 1000))
        print(test_data_dir)
        if not os.path.exists(test_data_dir):
            os.mkdir(test_data_dir)

        dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True)
        for s in opt.splits:
            start_time = time.perf_counter()
            sounds = dataset['fold{}'.format(s)].item()['sounds']
            labels = dataset['fold{}'.format(s)].item()['labels']

            valGen = ValGenerator(sounds, labels, opt)
            valX, valY = valGen.get_data()

            print('{}/fold{}_test{}'.format(test_data_dir, s, opt.batchSize))
            np.savez_compressed('{}/fold{}_test{}'.format(test_data_dir, s, opt.batchSize), x=valX, y=valY)
            print('split-{} test with shape x{} and y{} took {:.2f} secs'.format(s, valX.shape, valY.shape,
                                                                                 time.perf_counter() - start_time))
            sys.stdout.flush()
