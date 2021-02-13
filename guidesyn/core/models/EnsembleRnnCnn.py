"""
Copyright 2019 Software Reliability Lab, ETH Zurich
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
import torch
import torch.nn.functional as f
import numpy as np
from PIL import Image
import torch.optim as optim
import torchvision.utils as vutils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
from guidesyn.core.arguments import Data

# parameters CNN
filterConv1 = 64
filterConv2 = 32
filterConv3 = 16
dense_layer = 256

# parameters RNN
hidden_dim1 = 128
hidden_dim2 = 32

multiplier = 2
bidirectional = True

if bidirectional:
    multiplier = multiplier * 2


# consisting of RNN, CNN
class EnsembleRnnCnn(nn.Module):
    def __init__(self, args):
        super(EnsembleRnnCnn, self).__init__()
        self.name = "EnsembleRnnCnn"
        _height = args.target_image_height
        _width = args.target_image_width
        self.targetSize = int((_width / 8) * (_height / 8))

        self.lstm1 = nn.LSTM(hidden_dim2 * multiplier, hidden_dim1, bidirectional=bidirectional, batch_first=True)
        self.lstm2 = nn.LSTM(args.input_size, hidden_dim2, bidirectional=bidirectional, batch_first=True)

        self.conv1 = nn.Conv2d(1, filterConv1, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(filterConv1, filterConv2, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(filterConv2, filterConv3, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.fc1 = nn.Linear(filterConv3 * self.targetSize + hidden_dim1 * multiplier, dense_layer)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(dense_layer, args.outputClasses)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.args = args

    def forward(self, x, _input, batch_info):
        (data, lengths) = _input
        # RNN
        view_info = pack_padded_sequence(data, lengths.cpu(), batch_first=True)
        _, hidden_ind_view = self.lstm2(view_info)
        hidden_concat = torch.cat(hidden_ind_view, dim=-1)
        hidden_concat = hidden_concat.permute(1, 0, 2).contiguous()
        hidden_concat = hidden_concat.view(hidden_concat.size(0), -1)
        slices = torch.split(hidden_concat, batch_info.tolist(), 0)
        _pack_sequence = pack_sequence(list(slices))
        _, hidden = self.lstm1(_pack_sequence)
        hidden = torch.cat(hidden, dim=-1)
        hidden = hidden.permute(1, 0, 2).contiguous()
        output_rnn = hidden.view(hidden.size(0), -1)

        # CNN
        x = self.args.activation_function(self.conv1(x))
        x = self.pool1(x)
        x = self.args.activation_function(self.conv2(x))
        x = self.pool2(x)
        x = self.args.activation_function(self.conv3(x))
        x = self.pool3(x)
        x = x.view(-1, filterConv3 * self.targetSize)
        x = torch.cat((x, output_rnn), 1)

        x = self.args.activation_function(self.fc1(x))
        x = f.dropout(x, p=self.args.dropout_rate, training=self.training)
        x = self.fc2(x)
        assert len(x.shape) == self.args.outputClasses
        return x
