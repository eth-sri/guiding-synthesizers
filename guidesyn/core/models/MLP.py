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

intermediate = 512


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.name = "MLP"
        self.linear = nn.Linear(args.input_size, intermediate)
        self.linear_1 = nn.Linear(intermediate, intermediate)
        self.linear_3 = nn.Linear(intermediate, intermediate)
        self.linear_2 = nn.Linear(intermediate, args.outputClasses)
        self.args = args

    def forward(self, x):
        x = self.args.activation_function(self.linear(x))
        x = self.args.activation_function(self.linear_1(x))
        x = self.args.activation_function(self.linear_3(x))
        out = self.linear_2(x)
        return out
