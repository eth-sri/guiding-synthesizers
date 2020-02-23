"""
Copyright 2019 Secure, Reliable, and Intelligent Systems Lab, ETH Zurich
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
# from tensorboardX import SummaryWriter
from enum import Enum
import numpy as np
import torch
import os


class Data(Enum):
    BOTH = 1
    MANUAL = 2
    IMAGE = 3
    PROGRAM = 4
    IOP = 5


class DataModeRNN(Enum):
    UNUSED = 0
    FULL = 1
    RAW = 2
    ABSTRACT = 3


class ManualType(Enum):
    UNUSED = 0
    # multiple screen
    Manual_MLP_MS = 1
    # one screen
    Manual_MLP_OS = 2
    Manual_RNN_MS = 3
    Manual_RNN_OS = 4


def data_to_string(data_type):
    if data_type == Data.IMAGE:
        return "image"
    elif data_type == Data.MANUAL:
        return "manual"
    else:
        return "image+manual"


def manual_type_to_string(manual_type):
    return str(manual_type).replace("ManualType.", "").lower()


def data_mode_rnn_to_string(data_mode):
    return str(data_mode).replace("DataModeRNN.", "").lower()


def downsample_for_dimensions(image_width):
    return 4.0


class ArgumentsSubset:
    def __init__(self):
        pass


class Arguments:
    def __init__(self, name, dataset, data_type, manual_type, data_mode_rnn):
        self.datasetName = dataset
        self.modelName = name
        self.dataPrefixPath = "./data/rico/"

        self.batch_size = 10
        self.epochs = 5
        self.lr = 0.0001
        self.dropout_rate = 0.4

        self.save_model = False
        self.load_model = True

        # upper bound: playstore: 241, 516 -> 384. 640, rendered rico: 350, 630 -> 370, 650
        self.target_image_width = 392
        self.target_image_height = 656
        self._in_channel = 1
        self.outputClasses = 2
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 16]))
        self.activation_function = nn.ReLU()
        self.evaluationSize = 5000
        self.input_size = -1  # input_size of raw coordinates layer

        self.batch_norm = False
        self.use_gpu = True
        self.seed = 42

        self.data_type = data_type

        self.log_image_interval = 100
        self.log_net_out_input_interval = 100
        self.log_model_parameters = 200
        self.writeTensorboardX = False
        self.print_interval = 300
        self.manualDataType = manual_type
        self.manualFeatureType = data_mode_rnn
        self.normalized = False
        self.pixToDp = 1  # how coordinates will be adapted (pixel to dp) -> that margins match
        # self.time = 0 # for performance updates
        self.epoch_start = 1

    def print(self):
        print("Arguments:")
        print("dataset:", self.datasetName)
        print("modelName:", self.modelName)
        print("epochs", self.epochs)
        print("batch_size:", self.batch_size)
        print("learning_rate", self.lr)
        print("dropout_rate:", self.dropout_rate)
        print("target_image_width:", self.target_image_width)
        print("target_image_height:", self.target_image_height)
        print("evaluationSize:", self.evaluationSize)
        print("input_size:", self.input_size)
        print("load_model:", self.load_model)
        print("save_model: ", self.save_model)
        print("datatype:", self.data_type)
        print("manualDataType:", self.manualDataType)
        print("manualFeatureType:", self.manualFeatureType)
        print("normalized:", self.normalized)
        print("batch_norm: ", self.batch_norm)
        print("seed:", self.seed)
        print("save model path:", self.save_path)

    def to_small(self):
        new = ArgumentsSubset()
        new.datasetName = self.datasetName
        new.modelName = self.modelName
        new.data_type = self.data_type
        new.target_image_width = self.target_image_width
        new.target_image_height = self.target_image_height
        new.downsample = self.downsample
        new.manualFeatureType = self.manualFeatureType
        new.pixToDp = self.pixToDp
        new.extraOriginal = self.extraOriginal
        new.manualDataType = self.manualDataType
        new._in_channel = self._in_channel
        return new


def ds_dataset(dataset):
    return "ds" in dataset and "dsplus" not in dataset


def get_input_size(manual_type, data_mode_rnn):
    if manual_type == ManualType.Manual_MLP_MS:
        return 90
    elif manual_type == ManualType.Manual_MLP_OS:
        return 30
    elif manual_type == ManualType.Manual_RNN_MS:
        if data_mode_rnn == DataModeRNN.ABSTRACT:
            return 12
        elif data_mode_rnn == DataModeRNN.RAW:
            return 22
        else:
            return 34
    elif manual_type == ManualType.Manual_RNN_OS:
        if data_mode_rnn == DataModeRNN.RAW:
            return 11
        elif data_mode_rnn == DataModeRNN.ABSTRACT:
            return 17
        else:
            return 28
    else:
        print("Manual type not found, " + manual_type_to_string(manual_type) + " " + data_mode_rnn_to_string(
            data_mode_rnn))


def add_arguments_for_model(model, args):
    if args.data_type != Data.IMAGE:
        args.input_size = get_input_size(args.manualDataType, args.manualFeatureType)

    if model == "MLP":
        args.batch_size = 20
        args.lr = 0.001
        args.data_type = Data.MANUAL
    if model == "RNN":
        args.device = None
        args.batch_size = 256
        args.lr = 0.001
    if model == "CNN":
        args.batch_size = 15  # 18
        args.lr = 0.00001
    if model == "EnsembleRnnCnn":
        args.device = None
        args.data_type = Data.BOTH
        args.batch_size = 15
        args.lr = 0.0001


def add_epochs_more_data_experiment(model, args, dataset):
    args.save_model = False
    if model == "MLP":
        args.epochs = 5
    if model == "RNN":
        args.epochs = 20
    if model == "CNN":
        args.epochs = 20
    if model == "EnsembleRnnCnn":
        args.epochs = 25

    if "250" in dataset:
        args.epochs = args.epochs * 8
    elif "500" in dataset:
        args.epochs = args.epochs * 6
    elif "2000" in dataset:
        args.epochs = args.epochs * 4
    elif "8000" in dataset:
        args.epochs = args.epochs * 2

    if model == "MLP":
        args.epochs = 5


def add_epochs_oracle_experiment(model, args, dataset):
    if model == "MLP":
        args.epochs = 12
        if args.manualDataType == ManualType.Manual_MLP_MS:
            args.epochs = 14
        if ds_dataset(dataset):
            args.epochs = 4
    if model == "RNN":
        args.epochs = 70
        if ds_dataset(dataset):
            if args.manualFeatureType == DataModeRNN.FULL:
                args.epochs = 120
            elif args.manualFeatureType == DataModeRNN.RAW:
                args.epochs = 70
            else:
                args.epochs = 25
    if model == "CNN":
        args.epochs = 60
        if ds_dataset(dataset):
            args.epochs = 45
    if model == "EnsembleRnnCnn":
        args.epochs = 80
        if ds_dataset(dataset):
            args.epochs = 50


def add_data_path_for_dataset(dataset, args):
    if "ds" == dataset:
        args.dataPrefixPath = "./dataset/data/ds/"
    if "dsmoredata" in dataset:
        args.dataPrefixPath = "./dataset/data/dsmoredata-" + dataset.split("-")[1] + "/"
    if "dsplus" == dataset:
        args.dataPrefixPath = "./dataset/data/dsplus/"
    if "paperdsplusNS" == dataset:
        args.dataPrefixPath = "./dataset/data/dsplusNS/"
    if "paperdsplusFull" == dataset:
        args.dataPrefixPath = "./dataset/data/dsplusFULL/"


def update_arguments_for_dataset(dataset, args):
    if ds_dataset(dataset):
        args.extraOriginal = False
        args.pixToDp = 1
        args.downsample = 4
        args.shouldPredict = False
    elif "dsplus" in dataset or dataset == "dpp" or dataset == "dpg":
        args.extraOriginal = True
        args.pixToDp = 1
        args.downsample = 4
        args.shouldPredict = False
    elif "ablation" in dataset:  # ablation dataset
        args.extraOriginal = True
        args.pixToDp = 1
        args.downsample = 4
        args.shouldPredict = False
    else:  # if(dataset == "rico4" or dataset == "rico"):
        args.downsample = 4
        args.extraOriginal = False
        args.pixToDp = 4


def get_arguments(model_name, dataset, data_type, manual_type, data_mode_rnn, experiment, seed, model_path,
                  should_cache):
    args = Arguments(model_name, dataset, data_type, manual_type, data_mode_rnn)
    args.seed = seed
    args.cache = should_cache
    args.pretraining = "np"
    add_arguments_for_model(model_name, args)
    if experiment == "rico-moreData":
        add_epochs_more_data_experiment(model_name, args, dataset)
    else:
        add_epochs_oracle_experiment(model_name, args, dataset)

    add_data_path_for_dataset(dataset, args)
    update_arguments_for_dataset(dataset, args)

    seed_append = ""
    if int(seed) != 42:
        seed_append = "_" + str(args.seed) + "_"
    prefix_save_path = model_path
    args.save_path = prefix_save_path + args.modelName + "_" + args.datasetName + "_" + data_to_string(args.data_type) + \
                     np.format_float_positional(args.lr).split(".")[1] + "_" + str(
        args.batch_size) + "_" + args.pretraining + "_" + manual_type_to_string(
        args.manualDataType) + "_" + data_mode_rnn_to_string(args.manualFeatureType) + seed_append + ".pt"
    return args
