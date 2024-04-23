import os
import random
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'common'))

import common.utils as u


class Generator:
    # Generates data for Keras
    def __init__(self, samples, labels, options):
        random.seed(42)
        # Initialization
        self.data = [(samples[i], labels[i]) for i in range(0, len(samples))]
        self.opt = options
        self.batch_size = options.batchSize
        self.preprocess_funcs = self.preprocess_setup()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.data) / self.batch_size))
        # return len(self.samples)

    def __get_items__(self, batches):
        device = torch.device("cuda:0")
        batchesX = []
        batchesY = []

        for i in range (batches):
            batchX, batchY = self.__getitem__(i)
            batchesX.append(batchX)
            batchesY.append(batchY)

        return torch.stack(batchesX).to(device), torch.stack(batchesY).to(device)

    def __getitem__(self, batchIndex):
        # Generate one batch of data
        batchX, batchY = self.generate_batch(batchIndex)
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)

        return torch.tensor(np.moveaxis(batchX, 3, 1)), torch.tensor(batchY)

    def generate_batch(self, batchIndex):
        # Generates data containing batch_size samples
        sounds = []
        labels = []

        selected = []

        if self.opt.mixupFactor == 2:
            for i in range(self.batch_size):
                # Training phase of BC learning
                # Select two training examples
                while True:
                    ind1 = random.randint(0, len(self.data) - 1)
                    ind2 = random.randint(0, len(self.data) - 1)

                    sound1, label1 = self.data[ind1]
                    sound2, label2 = self.data[ind2]

                    if len({label1, label2}) == 2 and "{}-{}".format(ind1, ind2) not in selected:
                        selected.append("{}-{}".format(ind1, ind2))
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)

                # Mix two examples
                r = np.array(random.random())
                sound = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                eye = np.eye(self.opt.nClasses[self.opt.dataset])
                label = (eye[label1 - 1] * r + eye[label2 - 1] * (1 - r) ).astype(np.float32)

                # For stronger augmentation
                sound = u.random_gain(6)(sound).astype(np.float32)

                sounds.append(sound)
                labels.append(label)
        elif self.opt.mixupFactor == 3:
            for i in range(self.batch_size):
                # Training phase of BC learning
                # Select two training examples
                while True:
                    ind1 = random.randint(0, len(self.data) - 1)
                    ind2 = random.randint(0, len(self.data) - 1)
                    ind3 = random.randint(0, len(self.data) - 1)

                    sound1, label1 = self.data[ind1]
                    sound2, label2 = self.data[ind2]
                    sound3, label3 = self.data[ind3]

                    if len({label1, label2, label3}) == 3 and "{}-{}-{}".format(ind1, ind2, ind3) not in selected:
                        selected.append("{}-{}-{}".format(ind1, ind2, ind3))
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)
                sound3 = self.preprocess(sound3)

                # Mix three examples
                r = np.array(random.random())
                q = np.array(random.random())
                mix_sound = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                sound = u.mix(mix_sound, sound3, q, self.opt.sr).astype(np.float32)
                eye = np.eye(self.opt.nClasses[self.opt.dataset])
                label = (eye[label1 - 1] * r * q + eye[label2 - 1] * (1 - r) * q + eye[label3 - 1] * (1 - q)).astype(
                    np.float32)

                # For stronger augmentation
                sound = u.random_gain(6)(sound).astype(np.float32)

                sounds.append(sound)
                labels.append(label)
        elif self.opt.mixupFactor == 4:
            for i in range(self.batch_size):
                # Training phase of BC learning
                # Select two training examples
                while True:
                    ind1 = random.randint(0, len(self.data) - 1)
                    ind2 = random.randint(0, len(self.data) - 1)
                    ind3 = random.randint(0, len(self.data) - 1)
                    ind4 = random.randint(0, len(self.data) - 1)

                    sound1, label1 = self.data[ind1]
                    sound2, label2 = self.data[ind2]
                    sound3, label3 = self.data[ind3]
                    sound4, label4 = self.data[ind4]

                    if len({label1, label2, label3, label4}) == 4 and "{}-{}-{}-{}".format(ind1, ind2, ind3, ind4) not in selected:
                        selected.append("{}-{}-{}-{}".format(ind1, ind2, ind3, ind4))
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)
                sound3 = self.preprocess(sound3)
                sound4 = self.preprocess(sound4)

                # Mix four examples
                r = np.array(random.random())
                q = np.array(random.random())
                p = np.array(random.random())
                mix_sound1 = u.mix(sound1, sound2, r, self.opt.sr).astype(np.float32)
                mix_sound2 = u.mix(mix_sound1, sound3, q, self.opt.sr).astype(np.float32)
                sound = u.mix(mix_sound2, sound4, p, self.opt.sr).astype(np.float32)
                eye = np.eye(self.opt.nClasses[self.opt.dataset])
                label = (eye[label1 - 1] * r * q * p + eye[label2 - 1] * (1 - r) * q * p + eye[label3 - 1] * (1 - q) * p + eye[label4 - 1] * (1 - p)).astype(
                    np.float32)

                # For stronger augmentation
                sound = u.random_gain(6)(sound).astype(np.float32)

                sounds.append(sound)
                labels.append(label)

        sounds = np.asarray(sounds)
        labels = np.asarray(labels)

        return sounds, labels

    def preprocess_setup(self):
        funcs = []
        if self.opt.strongAugment:
            funcs += [u.random_scale(1.25)]

        funcs += [u.padding(self.opt.inputLength // 2),
                  u.random_crop(self.opt.inputLength),
                  u.normalize(32768.0)]
        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound

def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.sr // 1000)), allow_pickle=True)
    train_sounds = []
    train_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i != split:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    trainGen = Generator(train_sounds, train_labels, opt)
    print("* {} data ready to train the model".format(len(train_sounds)))
    return trainGen
