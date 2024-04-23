import torch
import torch.nn as nn
import numpy as np
import random

# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ESCNAS(nn.Module):
    def __init__(self, input_length, n_class, sr, ch_conf=None):
        super(ESCNAS, self).__init__()
        self.input_length = input_length
        self.ch_config = ch_conf

        stride1 = 2
        stride2 = 2
        channels = 16
        k_size = (3, 3)
        n_frames = (sr / 1000) * 10  # No of frames per 10ms

        sfeb_pool_size = int(n_frames / (stride1 * stride2))

        if self.ch_config is None:
            self.ch_config = [8, 64, 32, 32, 48, 60, 68, n_class]  # k_16_c_4
            # self.ch_config = [8, 64, 32, 32, 48, n_class]  # k_16_c_2

        fcn_no_of_inputs = self.ch_config[-1]

        # ACFE - Audio Contextual Feature Extractor
        conv1, bn1 = self.make_layers(1, self.ch_config[0], (1, 9), (1, stride1))
        conv2, bn2 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 5), (1, stride2))

        # TDFE - Time Dependent Feature Extractor
        conv3, bn3 = self.make_layers(1, self.ch_config[2], k_size, padding=1)

        # c = 4
        conv4, bn4 = self.make_layers(self.ch_config[2], self.ch_config[3], k_size, padding=1)
        conv5, bn5 = self.make_layers(self.ch_config[3], self.ch_config[4], k_size, padding=1)
        conv6, bn6 = self.make_layers(self.ch_config[4], self.ch_config[5], k_size, padding=1)
        conv7, bn7 = self.make_layers(self.ch_config[5], self.ch_config[6], k_size, padding=1)

        conv8, bn8 = self.make_layers(self.ch_config[6], self.ch_config[7], (1, 1))

        fcn = nn.Linear(fcn_no_of_inputs, n_class)

        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid')

        self.acfe = nn.Sequential(
            conv1, bn1, nn.ReLU(),
            conv2, bn2, nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )

        tdfe_modules = []

        tdfe_modules.extend([conv3, bn3, nn.ReLU()])
        tdfe_modules.append(nn.MaxPool2d(kernel_size=(2, 2)))

        # c = 4
        tdfe_modules.extend([conv4, bn4, nn.ReLU()])
        tdfe_modules.append(nn.MaxPool2d(kernel_size=(2, 2)))

        tdfe_modules.extend([conv5, bn5, nn.ReLU()])
        tdfe_modules.append(nn.MaxPool2d(kernel_size=(2, 2)))

        tdfe_modules.extend([conv6, bn6, nn.ReLU()])
        tdfe_modules.append(nn.MaxPool2d(kernel_size=(2, 2)))

        tdfe_modules.extend([conv7, bn7, nn.ReLU()])
        tdfe_modules.append(nn.MaxPool2d(kernel_size=(2, 2)))

        self.tdfe_width = int(((self.input_length / sr) * 1000) / 10)  # 10ms frames of audio length in seconds
        tdfe_pool_sizes = self.get_tdfe_pool_sizes(self.ch_config[1], self.tdfe_width)

        tdfe_modules.append(nn.Dropout(0.2))
        tdfe_modules.extend([conv8, bn8, nn.ReLU()])

        # h, w = 8, 18  # c_2
        # h, w = 4, 9  # c_3
        h, w = 2, 4  # c_4

        tdfe_modules.append(nn.AvgPool2d(kernel_size=(h, w)))
        tdfe_modules.extend([nn.Flatten(), fcn])

        self.tdfe = nn.Sequential(*tdfe_modules)

        self.output = nn.Sequential(
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.acfe(x)
        # swap axes
        x = x.permute((0, 2, 1, 3))
        x = self.tdfe(x)
        # x = x.mean([2, 3])  # global average pooling
        y = self.output[0](x)
        return y

    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn

    def get_tdfe_pool_sizes(self, con2_ch, width):
        h = self.get_tdfe_pool_size_component(con2_ch)
        w = self.get_tdfe_pool_size_component(width)
        pool_size = []

        for (h1, w1) in zip(h, w):
            pool_size.append((h1, w1))

        return pool_size

    def get_tdfe_pool_size_component(self, length):
        c = []
        index = 1

        while index <= 6:
            if length >= 2:
                if index == 6:
                    c.append(length)
                else:
                    c.append(2)
                    length = length // 2
            else:
                c.append(1)

            index += 1

        return c


def GetESCNASModel(input_len=30225, nclass=26, sr=20000, channel_config=None):
    net = ESCNAS(input_len, nclass, sr, ch_conf=channel_config)
    return net
