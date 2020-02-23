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

from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as f
import torch.optim as optim
import torchvision.utils as vutils
from guidesyn.core.arguments import Data

filterConv1 = 64
filterConv2 = 32
filterConv3 = 16
dense_layer = 256


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.name = "CNN"
        _height = args.target_image_height
        _width = args.target_image_width
        self.targetSize = int((_width / 8) * (_height / 8))
        self.conv1 = nn.Conv2d(1, filterConv1, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(filterConv1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(filterConv1, filterConv2, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm2d(filterConv2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(filterConv2, filterConv3, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm2d(filterConv3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        if args.data_type == Data.BOTH:
            self.fc1 = nn.Linear(filterConv3 * self.targetSize + args.input_size, dense_layer)
        else:  # Image only
            self.fc1 = nn.Linear(filterConv3 * self.targetSize, dense_layer)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.dense1_bn = nn.BatchNorm1d(dense_layer)
        self.fc2 = nn.Linear(dense_layer, args.outputClasses)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        self.args = args

    def forward(self, x, y=None):
        x = self.args.activation_function(self.conv1(x))
        if self.args.batch_norm:
            x = self.conv1_bn(x)
        x = self.pool1(x)
        x = self.args.activation_function(self.conv2(x))
        if self.args.batch_norm:
            x = self.conv2_bn(x)
        x = self.pool2(x)
        x = self.args.activation_function(self.conv3(x))
        if self.args.batch_norm:
            x = self.conv3_bn(x)
        x = self.pool3(x)
        x = x.view(-1, filterConv3 * self.targetSize)
        if self.args.data_type == Data.BOTH:
            x = torch.cat((x, y), 1)
        x = self.fc1(x)
        # x = self.dense1_bn(x)
        x = self.args.activation_function(x)
        x = f.dropout(x, p=self.args.dropout_rate, training=self.training)
        x = self.fc2(x)
        assert len(x.shape) == self.args.outputClasses
        return x
