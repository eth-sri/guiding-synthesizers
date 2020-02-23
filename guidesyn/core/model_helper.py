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
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from guidesyn.core.arguments import Data, ds_dataset
from guidesyn.core.custom_data_reader import get_custom_dataset, read_views
from guidesyn.core.models.CNN import CNN
from guidesyn.core.models.RNN import RNN
from guidesyn.core.models.MLP import MLP
from guidesyn.core.models.EnsembleRnnCnn import EnsembleRnnCnn


def model_for_name(args, device):
    if args.modelName == "CNN":
        model = CNN(args).to(device)
    if args.modelName == "MLP":
        model = MLP(args).to(device)
    if args.modelName == "RNN":
        model = RNN(args)
    if args.modelName == "EnsembleRnnCnn":
        model = EnsembleRnnCnn(args)
    try:  # trick to avoid cudnn error on drlszlarge (addressing issue: https://github.com/pytorch/pytorch/issues/17543)
        model.to(device)
    except:
        model.to(device)
    return model


def get_model_output(args, sample_batched, model, device):
    if args.data_type == Data.MANUAL:
        if args.modelName == "RNN":
            data = (sample_batched['manual'][0].to(device), sample_batched['manual'][1].to(device))
            return model(data, sample_batched['batch_counter'].to(device))
        else:
            data = sample_batched['manual'].to(device)
        return model(data)
    elif args.data_type == Data.IMAGE:
        data = sample_batched['image'].to(device)
        return model(data)
    elif args.modelName == "EnsembleRnnCnn":
        image_data = sample_batched['image'].to(device)
        data = (sample_batched['manual'][0].to(device), sample_batched['manual'][1].to(device))
        return model(image_data, data, sample_batched['batch_counter'].to(device))
    else:
        print("Did not find model")


def train(args, model, device, dataset_length, optimizer, epoch, batch_idx, sample_batched, train_step):
    model.train()
    target = sample_batched['label'].to(device)
    optimizer.zero_grad()
    output = get_model_output(args, sample_batched, model, device)

    # output, target
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = float(correct) / float(len(target))
    loss = args.criterion(output, target)

    loss.backward()
    optimizer.step()
    if (batch_idx + 1) % args.print_interval == 0:
        print('{},{}: Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}, ({}/{})'.format(
            args.modelName, args.datasetName,
            epoch, batch_idx * len(sample_batched['label']), dataset_length * args.batch_size,
                   100. * batch_idx / dataset_length, loss.item(), acc * 100., correct, len(target)))

    return correct, len(target)


def precision(tp, fp):
    if tp + fp == 0:
        return -1
    return float(tp) / float(tp + fp)


def recall(tp, fn):
    if tp + fn == 0:
        return -1
    return float(tp) / float(tp + fn)


def test(args, model, device, test_loader, maximum_steps, log_step, current_epoch):
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    with torch.no_grad():
        for sample_batched in test_loader:
            output = get_model_output(args, sample_batched, model, device)
            target = sample_batched['label'].to(device)
            args.criterion.reduction = 'sum'
            loss = args.criterion(output, target)
            test_loss += loss.float()
            args.criterion.reduction = 'mean'
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            pred_filtered = pred.clone()
            pred_filtered[pred_filtered == 0] = -1
            true_pos = pred_filtered.eq(target.view_as(pred)).sum().item()
            false_positives += (pred.nonzero().size(0) - true_pos)
            true_positives += true_pos
            false_negatives += (target.nonzero().size(0) - true_pos)
            correct += pred.eq(target.view_as(pred)).sum().item()
            counter = counter + len(target)
            if (maximum_steps != -1) and (counter > maximum_steps):
                break

    print("False positives: " + str(false_positives))
    print("True positives:" + str(true_positives))
    print("False negatives:" + str(false_negatives))
    true_negative = counter - (true_positives + false_positives + false_negatives)
    print("True negatives:" + str(true_negative))
    steps = counter
    if maximum_steps == -1:
        steps = len(test_loader.dataset)

    test_loss /= steps
    print(
        '{}, {}\nTest set (Epoch {}): Average loss: {:.4f}, Recall: ({:.0f}%), Precision: ({:.0f}%), Accuracy: {}/{} ({:.0f}%)\n'.format(
            args.modelName, args.datasetName, current_epoch,
            test_loss, 100. * recall(true_positives, false_negatives),
                       100. * precision(true_positives, false_positives), correct, steps,
                       100. * correct / steps))
    return 100. * correct / steps, true_positives, false_positives, false_negatives, true_negative, test_loss


def predict(args, model, device, data):
    model.eval()
    with torch.no_grad():
        output = get_model_output(args, data, model, device)
        if isinstance(args.criterion, nn.CrossEntropyLoss):
            output = F.softmax(output)
        elif isinstance(args.criterion, nn.NLLLoss):
            output = torch.exp(output)
        pred = output.max(1, keepdim=True)[1]
        return int(pred.data[0])
    return -1


def good_file(bad_name, root_dir, args):
    if not args.extraOriginal:
        return True, bad_name.split("_")[0] + "_1.txt"
    # max 30 view candidates
    for i in range(0, 30):
        name = bad_name.split("-")[0] + "-" + bad_name.split("-")[1] + "-" + bad_name.split("-")[2] + "-" + str(
            i) + "_1.txt"
        if os.path.isfile(os.path.join(root_dir, name)):
            return True, name
    name = bad_name.split("-")[0] + "-" + bad_name.split("-")[1] + "-" + bad_name.split("-")[2] + "-tr_1.txt"
    if os.path.isfile(os.path.join(root_dir, name)):
        return True, name
    return False, "Does not exist"


def predict_batch(args, model, device, dataset, desc=None, max_samples=-1):
    model.eval()
    file_list_neg = [s for s in os.listdir(dataset) if ("_0.txt" in s)]

    if ds_dataset(dataset):
        file_list_neg.sort(key=lambda x: len(open(dataset + x).readlines()), reverse=True) # my_collate_rnn
    else:
        file_list_neg.sort(key=lambda x: int(x.split("-")[2]), reverse=True) # my_collate_rnn

    file_list_pos = [good_file(s, dataset, args)[1] for s in file_list_neg]

    data_loader_evaluation_pos = get_custom_dataset(args, dataset, False, True, file_list_pos)
    data_loader_evaluation_neg = get_custom_dataset(args, dataset, False, True, file_list_neg)

    total = 0
    it_neg = iter(data_loader_evaluation_neg)
    correct_classified = 0
    correct_classified_se = 0
    for i, sample_batched in enumerate(tqdm(data_loader_evaluation_pos, desc=desc, disable=(desc is None))):
        if max_samples != -1 and i * len(sample_batched["label"]) > max_samples:
            break
        output = get_model_output(args, sample_batched, model, device)
        batch_neg = next(it_neg)
        output_neg = get_model_output(args, batch_neg, model, device)
        correct_classified = correct_classified + (output_neg[:, 1] < output[:, 1]).nonzero().size(
            0)
        correct_classified_se = correct_classified_se + (output_neg[:, 1] <= output[:, 1]).nonzero().size(
            0)
        total += len(output)
    return 100. * correct_classified / total, 100. * correct_classified_se / total, correct_classified, total


# add caching of the generated negative file list
negListCache = {}


def get_negative_list(len_pos_scores, file_list_pos, dataset, args, maximum):
    _id = dataset + str(maximum)
    if _id in negListCache:
        return negListCache[_id]

    filelist_counter = []
    total_file_list_neg = list()
    for i in range(0, len_pos_scores):
        f = file_list_pos[i]
        if args.extraOriginal:
            file_list_neg = [s for s in os.listdir(dataset) if
                             ("_0.txt" in s) and s.split("-")[0] == f.split("-")[0] and s.split("-")[1] == f.split("-")[
                                 1] and s.split("-")[2] == f.split("-")[2]]
        else:
            file_list_neg = [s for s in os.listdir(dataset) if ("_0.txt" in s) and f.split("_")[0] == s.split("_")[0]]
        filelist_counter.append(len(file_list_neg))
        total_file_list_neg.extend(file_list_neg)
        if maximum != -1 and i >= len_pos_scores:
            break
    negListCache[_id] = (total_file_list_neg, filelist_counter)
    return negListCache[_id]


# cache filelist
def predict_one_out_of_many(args, model, device, dataset, maximum=-1, desc=None):
    model.eval()
    file_list_pos = [s for s in os.listdir(dataset) if ("_1.txt" in s)]

    if ds_dataset(dataset):
        file_list_pos.sort(key=lambda x: len(open(dataset + x).readlines()), reverse=True)  # my_collate_rnn
    else:
        file_list_pos.sort(key=lambda x: int(x.split("-")[2]), reverse=True)  # my_collate_rnn

    pos_scores = list()
    data_loader_evaluation_pos = get_custom_dataset(args, dataset, False, True, file_list_pos)

    for batch_idx, sample_batched in enumerate(tqdm(data_loader_evaluation_pos, desc=desc, disable=(desc is None))):
        output = F.softmax(get_model_output(args, sample_batched, model, device), dim=1)
        pos_scores.extend(output[:, 1].tolist())
        if maximum != -1 and args.batch_size * batch_idx > maximum:
            break

    correct = 0
    se_correct = 0
    total = 0
    (totalFileListNeg, filelist_counter) = get_negative_list(len(pos_scores), file_list_pos, dataset, args, maximum)

    assert (len(filelist_counter) == len(pos_scores))
    data_loader_evaluation_neg = get_custom_dataset(args, dataset, False, True, totalFileListNeg)

    neg_scores = list()
    for sample_batched in data_loader_evaluation_neg:
        output = F.softmax(get_model_output(args, sample_batched, model, device), dim=1)
        neg_scores.extend(output[:, 1].tolist())

    start = 0
    for i, pos_score in enumerate(pos_scores):
        if maximum != -1 and i >= maximum:
            break
        s_correct = True
        se_t_correct = True
        subsamples = neg_scores[start: (start + filelist_counter[i])]
        if len(subsamples) > 0:
            if max(subsamples) >= pos_score:
                s_correct = False
            if max(subsamples) > pos_score:
                se_t_correct = False
            start = start + filelist_counter[i]
        # else:
        # print("Only positive candidate: ", dataset, fileListPos[i], maximum, len(fileListPos), start, start + filelistCounter[i])
        total = total + 1
        correct = correct + int(s_correct)
        se_correct = se_correct + int(se_t_correct)
    return 100. * correct / total, 100. * se_correct / total, total
