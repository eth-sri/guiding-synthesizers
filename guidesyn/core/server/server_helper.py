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
from flask import request
from flask import jsonify
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from PIL import Image, ImageDraw
from os import listdir
from os.path import isfile, join
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
import copy
import numpy as np
import time
from guidesyn.core.arguments import get_arguments, ManualType, DataModeRNN, downsample_for_dimensions, ds_dataset, Data, \
    manual_type_to_string, data_mode_rnn_to_string
from guidesyn.core.view import View, Constraint
from guidesyn.core.custom_data_reader import get_manual_data, get_image_data
from guidesyn.core.model_helper import get_model_output, model_for_name

measurements = 0
total_time = 0

models = {}


def resetModel():
    global models
    models = {}


def mode_for_name(name):
    if "abs" in name.lower():
        return DataModeRNN.ABSTRACT
    if "raw" in name.lower():
        return DataModeRNN.RAW
    else:
        print("Name not found, default.")
        return DataModeRNN.FULL


def getSelectedServer(name, dataset):
    if name == "MLP":
        if ds_dataset(dataset):
            selected_server = ("MLP", dataset, Data.MANUAL, ManualType.Manual_MLP_OS, DataModeRNN.UNUSED, "")
        else:
            selected_server = ("MLP", dataset, Data.MANUAL, ManualType.Manual_MLP_MS, DataModeRNN.UNUSED, "")
    elif name == "CNN":
        selected_server = ("CNN", dataset, Data.IMAGE, ManualType.UNUSED, DataModeRNN.UNUSED, "")
    elif "ensembleRnnCnn" in name:
        if ds_dataset(dataset):
            selected_server = ("EnsembleRnnCnn", dataset, Data.BOTH, ManualType.Manual_RNN_OS, mode_for_name(name), "")
        else:
            selected_server = ("EnsembleRnnCnn", dataset, Data.BOTH, ManualType.Manual_RNN_MS, mode_for_name(name), "")
    elif "RNN" in name:
        if ds_dataset(dataset):
            selected_server = ("RNN", dataset, Data.MANUAL, ManualType.Manual_RNN_OS, mode_for_name(name), "")
        else:
            selected_server = ("RNN", dataset, Data.MANUAL, ManualType.Manual_RNN_MS, mode_for_name(name), "")
    else:
        print(name)
        print("Target not found")
    print("Selected server: " + selected_server[0])
    return selected_server


def model_for_key(name, selected_server, gpu_id, model_path):
    if name in models:
        return models[name]

    (modelName, dataset, dataType, manualType, dataModeRnn, experiment) = selected_server
    args = get_arguments(modelName, dataset, dataType, manualType, dataModeRnn, experiment, 42, model_path, False)
    use_cuda = args.use_gpu and torch.cuda.is_available()
    print("Cuda available: ", torch.cuda.is_available())

    if use_cuda and gpu_id != -1:
        #print('Available devices ', torch.cuda.device_count())
        device = torch.device("cuda:" + str(gpu_id))
    else:
        device = torch.device("cpu")

    model = model_for_name(args, device)
    print(model.name)

    print("Try to load model: " + args.save_path)
    if os.path.isfile(args.save_path):
        print("Loading model")
        print(args.save_path)
        checkpoint = torch.load(args.save_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Failed to load model")
        exit(0)
    models[name] = (model, args, device)
    return models[name]


def predict(model, args, device, data):
    model.eval()
    with torch.no_grad():
        output = get_model_output(args, data, model, device)
        if isinstance(args.criterion, nn.CrossEntropyLoss):
            output = F.softmax(output)
        elif isinstance(args.criterion, nn.NLLLoss):
            output = torch.exp(output)
        return output[0][1].item() # prediction true
    return -1


def transformBatchRnn(batch):
    _list = []
    batch_counter = []
    for d in batch:
        dic = d["manual"]
        batch_counter.append(dic.size(0))
        for i in range(0, dic.size(0)):
            _list.append(dic[i, :, :])
    _pack_sequence = pack_sequence(_list)
    manual_data = pad_packed_sequence(_pack_sequence, batch_first=True)
    # image is only correct if batch size here is always 1
    return {"manual": manual_data, "image": batch[0]["image"], "batch_counter": torch.tensor(batch_counter)}


def pixToDp(image_width):
    # if(int(image_width) == 1440):
    #    return 4.0
    # elif((int(image_width)) > 680):
    #    #playstore 682, 720, 768
    #    return 2.0
    # rendered rico 350, 360, 370
    # now we are expecting that all input data has the input size of ~ 1440 x 2560
    return 1.0


def getScore(views, original_views, model, args, device, pixToDpVal):
    number_of_views = len(views)
    image = []
    if args.data_type == Data.IMAGE or args.data_type == Data.BOTH:
        image_data = get_image_data(views, args, True)
        image = image_data[None, :, :, :]

    manual = []
    if args.data_type == Data.MANUAL or args.data_type == Data.BOTH or args.data_type == Data.IOP:
        views = copy.deepcopy(views)
        for view in views:
            view._pix_to_dp(pixToDpVal)

        original_views = copy.deepcopy(original_views)
        for org_view in original_views:
            org_view._pix_to_dp(pixToDpVal)

        manual = get_manual_data(args, views, original_views)
        if not (args.modelName == "RNN" or args.modelName == "EnsembleRnnCnn"):
            manual = manual[None, :]

    data = {'image': image, 'manual': manual}
    if number_of_views != len(views):
        print("Mistake: Number of views do not match")
    elif args.modelName == "RNN" or args.modelName == "EnsembleRnnCnn":
        data = transformBatchRnn([data])
    return predict(model, args, device, data)


def equal(views, target_views):
    for idx, val in enumerate(views):
        if not val.equal(target_views[idx]):
            return False
    return True


def print_app(views):
    for idx, val in enumerate(views):
        print(val.file_string())


def predict_example(content, model_path, gpu_id=0, selected_server=None, server_name="", dataset="", targetXML=None):
    if selected_server is None:
        selected_server = getSelectedServer(server_name, dataset)

    model_iden = selected_server[0] + selected_server[1] + manual_type_to_string(
        selected_server[3]) + data_mode_rnn_to_string(selected_server[4])
    (model, args, device) = model_for_key(model_iden, selected_server, gpu_id, model_path)

    dimensions = content["dimensions"]
    device_width = float(dimensions[0])
    device_height = float(dimensions[1])
    number_of_views = 0

    downsample = 1.0
    args.downsample = downsample_for_dimensions(device_width)
    # if (args.data_type == Data.IMAGE or args.data_type == Data.BOTH) and (device_height/downsample) != args.target_image_height:
    # print("Warning, image does not have same scale as training data set", downsample, (device_height/float(args.target_image_height)))
    # return jsonify(result = -1)

    original_views = []
    target_length = len(content["layouts"][0]["Views"])
    for org_views in content["original"]["Views"]:
        original_view = View(int(org_views['x']), int(org_views['y']), int(org_views['width']),
                             int(org_views['height']), 0.0, 0.0)
        # if targetXML is not None and "vertical" in targetXML and int(view_json['id']) != 0:
        # expect the layouts to be sorced -> int(view_json['id'])-1 is the position in the array
        # original_view.add_constraints(Constraint.constraintFromJSON(targetXML["vertical"][int(org_views['id'])-1]), Constraint.constraintFromJSON(targetXML["horizontal"][int(org_views['id'])-1]))
        original_views.append(original_view)
        if len(original_views) == target_length:
            break

    score = []
    apps = []
    layout_id = 0
    for layout in content["layouts"]:
        views = []
        for view_json in layout["Views"]:
            new_view = View(int(view_json['x']), int(view_json['y']), int(view_json['width']), int(view_json['height']),
                            0.0, 0.0)
            if 'vert_const' in view_json and 'hort_const' in view_json:
                new_view.add_constraints(Constraint.constraint_from_json(view_json['vert_const']),
                                         Constraint.constraint_from_json(view_json['hort_const']))
            views.append(new_view)

        number_of_views = len(views)
        apps.append(views)
        start = time.time()
        score.append(getScore(views, original_views, model, args, device, pixToDp(device_width)))
        end = time.time()
        global measurements
        global total_time
        measurements = measurements + 1
        total_time = total_time + (end - start)
        print("Time score: ", total_time, measurements)
        layout_id = layout_id + 1

    result_index = score.index(
        max(score))  # maxIndexWithTriebreaker(apps, score) #is done automatically by the order of the layouts
    print("Selected: " + str(result_index))

    target_views = []
    for view_json in content["target"]["Views"]:
        target_view = View(int(view_json['x']), int(view_json['y']), int(view_json['width']), int(view_json['height']),
                           0.0, 0.0)
        if targetXML is not None and "vertical" in targetXML and int(view_json['id']) != 0:
            target_view.add_constraints(
                Constraint.constraint_from_json(targetXML["vertical"][int(view_json['id']) - 1]),
                Constraint.constraint_from_json(targetXML["horizontal"][int(view_json['id']) - 1]))
        target_views.append(target_view)
        if len(target_views) == number_of_views:
            break

    true_id = -1

    for idx, val in enumerate(apps):
        if equal(val, target_views):
            true_id = idx

    if not equal(apps[result_index], target_views):
        print("Mismatch, true candidate in " + str(true_id))

    target_score = getScore(target_views, original_views, model, args, device, pixToDp(device_width))
    return apps, target_views, result_index, score, downsample, target_score, true_id, original_views
