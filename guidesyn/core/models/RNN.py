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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence

hidden_dim1 = 128
hidden_dim2 = 32
dense_layer = 512

multiplier = 2
bidirectional = True

if bidirectional:
    multiplier = multiplier * 2


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.name = "RNN"
        self.lstm2 = nn.LSTM(args.input_size, hidden_dim2, bidirectional=bidirectional, batch_first=True)
        self.lstm1 = nn.LSTM(hidden_dim2 * multiplier, hidden_dim1, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(hidden_dim1 * multiplier, dense_layer)
        self.linear1 = nn.Linear(dense_layer, 2)
        self.device = args.device
        self.args = args

    def forward(self, _input, batch_info):
        (x, lengths) = _input
        inputs = pack_padded_sequence(x, lengths, batch_first=True)
        _, hidden_ind_view = self.lstm2(inputs)
        # concat tuple
        hidden_concat = torch.cat(hidden_ind_view, dim=-1)
        # returns tuple, (num_layers * num_directions, batch, hidden_size)
        hidden_concat = hidden_concat.permute(1, 0, 2).contiguous()
        hidden_concat = hidden_concat.view(hidden_concat.size(0), -1)
        # split it back in earlier batch info
        slices = torch.split(hidden_concat, batch_info.tolist(), 0)
        _pack_sequence = pack_sequence(list(slices))
        _, hidden = self.lstm1(_pack_sequence)
        # split tuple
        hidden = torch.cat(hidden, dim=-1)
        hidden = hidden.permute(1, 0, 2).contiguous()
        hidden = hidden.view(hidden.size(0), -1)
        out = self.args.activation_function(self.linear(hidden))
        output = self.linear1(out)
        return output
