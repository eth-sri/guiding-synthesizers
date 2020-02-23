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

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.optim as optim
import os
import torchvision.utils as vutils
from torch.autograd import Variable
import time
from tqdm import tqdm
import pprint
from guidesyn.core.arguments import get_arguments, data_to_string, update_arguments_for_dataset, manual_type_to_string, \
    data_mode_rnn_to_string, Data, ds_dataset
from guidesyn.core.model_helper import train, test, predict, predict_batch, model_for_name, predict_one_out_of_many
from guidesyn.core.custom_data_reader import get_custom_dataset
from guidesyn.core.arguments import ds_dataset


def ablation_dataset():
    return ["./dataset/data/ablation_dataset/dsplus_perserve_aspect_ratio1-0/",
            "./dataset/data/ablation_dataset/dsplus_perserve_popular_aspect_ratios/",
            "./dataset/data/ablation_dataset/dsplus_perserve_centering_horizontally_one_view/",
            "./dataset/data/ablation_dataset/dsplus_perserve_centering_horizontally_views/",
            "./dataset/data/ablation_dataset/dsplus_perserve_centering_vertically_one_view/",
            "./dataset/data/ablation_dataset/dsplus_perserve_centering_vertically_views/",
            "./dataset/data/ablation_dataset/dsplus_perserve_inside_screeen/",
            "./dataset/data/ablation_dataset/dsplus_perserve_intersections/",
            "./dataset/data/ablation_dataset/dsplus_perserve_margin_0_horizontally/",
            "./dataset/data/ablation_dataset/dsplus_perserve_margin_0_vertically/",
            "./dataset/data/ablation_dataset/dsplus_perserve_margin_horizontally/",
            "./dataset/data/ablation_dataset/dsplus_perserve_margin_vertically/",
            "./dataset/data/ablation_dataset/dsplus_perserve_similar_dimensions/"]


def validation_dataset(dataset):
    if ds_dataset(dataset):
        # train on D_S: evaluate D_S, D_S+
        return ["./dataset/data/ds/validate/", "./dataset/data/dsplus/validate/"]
    if "dsplus" in dataset:
        # trained on D_S+ and evaluated on D_S+
        return ["./dataset/data/dsplus/validate/"]
    return []


def format_test_result(test_result):
    acc_string = '{:.2f}%'.format(test_result[0])
    return {"Acc:": acc_string, "tp": test_result[1], "fp": test_result[2], "fn": test_result[3], "tn": test_result[4]}


def format_train_result(correct, total, correct_epoch, total_epoch):
    per_last = -1
    per_accumulated = -1
    if total != 0:
        per_last = float(correct) / float(total)
    if total_epoch != 0:
        per_accumulated = float(correct_epoch) / float(total_epoch)
    last = '{:.2f}%,[{}/{}]'.format(100 * per_last, correct, total)
    accumulated = '{:.2f}%,[{}/{}]'.format(100 * per_accumulated, correct_epoch, total_epoch)
    return {"Train acc:": last, "Train acc accumulated:": accumulated}


def evaluate_datasets(args, model, device, descriptor, train_acc):
    datapath = args.dataPrefixPath
    val_one_many_res = predict_one_out_of_many(args, model, device, (datapath + "validate/"))
    train_one_many_res = predict_one_out_of_many(args, model, device, (datapath + "train/"), 200)
    one_out_of_many_accuracy = (
        '{}:val={:.2f}%,{:.2f}%,train={:.2f}%,{:.2f}%'.format(descriptor, val_one_many_res[0],
                                                              val_one_many_res[1],
                                                              train_one_many_res[0],
                                                              train_one_many_res[1]))
    eval_ranking_result = predict_batch(args, model, device, (datapath + "validate/"))
    train_ranking_result = predict_batch(args, model, device, (datapath + "train/"), max_samples=400)
    pairwise_accuracy = (
        '{}:val={:.2f}%,{:.2f}%,[{}/{}],train=,{:.2f}%,{:.2f}%,[{}/{}]'
            .format(descriptor, eval_ranking_result[0],
                    eval_ranking_result[1],
                    eval_ranking_result[2],
                    eval_ranking_result[3],
                    train_ranking_result[0],
                    train_ranking_result[1],
                    train_ranking_result[2],
                    train_ranking_result[3]))
    # uncomment if you are not interested in results on ds+ for models trained on ds
    if ds_dataset(args.dataPrefixPath):
        update_arguments_for_dataset("./dataset/data/dsplus/test/", args)
        datapath = datapath.replace("ds", "dsplus")
        val_one_many_res = predict_one_out_of_many(args, model, device, (datapath + "validate/"))
        one_out_of_many_accuracy += (",val-ds+={:.2f}%,{:.2f}%".format(val_one_many_res[0],
                                                                       val_one_many_res[1]))
        eval_ranking_result = predict_batch(args, model, device, (datapath + "validate/"))
        pairwise_accuracy += (",val-ds+={:.2f}%,{:.2f}%,[{}/{}]".format(eval_ranking_result[0],
                                                                        eval_ranking_result[1],
                                                                        eval_ranking_result[2],
                                                                        eval_ranking_result[3]))
        update_arguments_for_dataset("./dataset/data/ds/test/", args)
    return pairwise_accuracy, one_out_of_many_accuracy


def evaluate_test_dataset(args, model, device, descriptor, train_acc):
    test_one_many_res = predict_one_out_of_many(args, model, device, (args.dataPrefixPath + "test/"),
                                                desc="Compute accuracy one vs. many on {}".format(descriptor))
    test_accuracy_one_out_of_many = (
        '{}:test={:.2f}%,{:.2f}%'.format(descriptor, test_one_many_res[0], test_one_many_res[1]))
    test_pairwise_accuracy_res = predict_batch(args, model, device, (args.dataPrefixPath + "test/"),
                                               desc="Compute pairwise accuracy on {}".format(descriptor))
    test_pairwise_accuracy = (
        '{}:test=,{:.2f}%,{:.2f}%'.format(descriptor, test_pairwise_accuracy_res[0], test_pairwise_accuracy_res[1]))
    return test_pairwise_accuracy, test_accuracy_one_out_of_many


def save_model(args, epoch, model, optimizer, append=""):
    try:
        save_path_epoch = args.save_path.split(".pt")[0] + "_" + str(epoch) + ".pt"
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, save_path_epoch)
    except Exception as e:
        print("Error saving model")
        print(e)


def run_training(args, model, device, dataset_name, result_dict, optimizer):
    pairwise_accuracy = []
    one_out_of_many_accuracy = []
    args.print()
    data_loader_training = get_custom_dataset(args, (args.dataPrefixPath + "train/"), True, False)
    # for logging purposes
    train_step = 0
    for epoch in range(args.epoch_start, args.epochs + 1):
        correct_epoch = 0
        total_epoch = 0

        start_whole_epoch = time.time()
        time_per_epoch = 0
        evaluate_time_per_epoch = 0
        for batch_idx, sample_batched in enumerate(data_loader_training):
            start_batch_training = time.time()
            (correct, total) = train(args, model, device, len(data_loader_training), optimizer, epoch, batch_idx,
                                     sample_batched, train_step)
            end_batch_training = time.time()
            time_per_epoch = time_per_epoch + (end_batch_training - start_batch_training)

            correct_epoch = correct_epoch + correct
            total_epoch = total_epoch + total
            train_step = train_step + 1

            if batch_idx == int(len(data_loader_training) / 2):
                # also evaluate on the middle of the batch
                evaluate_time = time.time()
                eval_dataset_result = evaluate_datasets(args, model, device, str(epoch) + "half",
                                                        100 * float(correct_epoch) / float(total_epoch))
                evaluate_time_end = time.time()
                evaluate_time_per_epoch = evaluate_time_per_epoch + (evaluate_time_end - evaluate_time)
                pairwise_accuracy.append(eval_dataset_result[0])
                one_out_of_many_accuracy.append(eval_dataset_result[1])
                if args.save_model:
                    save_model(args, epoch, model, optimizer, "_half")

        div = 90
        # if(experimentType == "rico-moreData"):
        #    div = 30
        if epoch % (max(1, int(args.epochs / div))) == 0:  # int(args.epochs/10)) #10 and epoch % 10 == 0:
            start_second_half = time.time()
            eval_dataset_result = evaluate_datasets(args, model, device, epoch,
                                                    100 * float(correct_epoch) / float(total_epoch))
            pairwise_accuracy.append(eval_dataset_result[0])
            one_out_of_many_accuracy.append(eval_dataset_result[1])
            print("Model: {} ,(epoch = {}), \n 1many: {},\n ranking: {}".format(args.modelName, epoch,
                                                                                eval_dataset_result[0],
                                                                                eval_dataset_result[1]))
            results_dict = {}
            # for val_dataset in validation_dataset(dataset_name):
            #     data_name_temp = val_dataset.split("/")[3]
            #     update_arguments_for_dataset(data_name_temp, args)
            #     # accuracy_prediction = predict_batch(args, model, device, testDataset)
            #     val_one_many_res = predict_one_out_of_many(args, model, device, val_dataset)
            #     # results_dict[(data_name_temp + "-ranking")] = '{:.2f}%,{:.2f}%,[{}/{}]'.format(accuracy_prediction[0], accuracy_prediction[1], accuracy_prediction[2], accuracy_prediction[3])
            #     results_dict[(data_name_temp + "-val_oneVSMany")] = '{:.2f}%,{:.2f}%,[{}]'.format(val_one_many_res[0],
            #                                                                                   val_one_many_res[1],
            #                                                                                   val_one_many_res[2])
            update_arguments_for_dataset(dataset_name, args)
        result_dict[epoch] = (format_train_result(correct, total, correct_epoch, total_epoch), results_dict)
        if args.save_model:
            save_model(args, epoch, model, optimizer)
        end_second_half = time.time()
    end_whole_epoch = time.time()
    print(
        'Evaluating time: Model {}, e = {}, training time {:.2f}, one eval {:.2f}, second half {:.2f} per totalTime: {:.2f}'.format(
            args.modelName, epoch, time_per_epoch, evaluate_time_per_epoch, (end_second_half - start_second_half),
            (end_whole_epoch - start_whole_epoch)))


    result_dict['pairwise_accuracy'] = pairwise_accuracy
    # for better readability
    one_out_of_many_accuracy.append(args.modelName + "-" + args.datasetName)
    result_dict['one_out_of_many_accuracy'] = one_out_of_many_accuracy
    result_dict['val_one_out_of_many_accuracy-max'] = max(one_out_of_many_accuracy, key=lambda x: float(
        x.split("%")[0].split("val=")[1] if ("val=" in x) else 0.0))
    result_dict['val_pariwise_accuracy-max'] = max(pairwise_accuracy, key=lambda x: float(
        x.split("%")[0].split("val=")[1] if ("val=" in x) else 0.0))

    print("Eval result for:" + args.modelName + "-" + args.datasetName)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result_dict)


def evaluate(args, model, device, dataset_name, result_dict, optimizer):
    test_pairwise_accuracy = []
    test_one_of_many_accuracy = []
    if ds_dataset(dataset_name):
        eval_test_dataset_result_ds = evaluate_test_dataset(args, model, device, "ds", -1.0)
        # The results on ds are better since it is trained on ds (with these specific device dimensions)
        # with the modifications according to the synthesizer we tried to be as close as possible to the changes in ds+
        result_dict['test_pairwise_accuracy-ds'] = eval_test_dataset_result_ds[0]
        result_dict['test_accuracy_one_vs_many-ds'] = eval_test_dataset_result_ds[1]

        # Evaluate ds dataset on ds+
        update_arguments_for_dataset("./dataset/data/dsplus/test/", args)
        args.dataPrefixPath = "./dataset/data/dsplus/"
        eval_test_dataset_result = evaluate_test_dataset(args, model, device, "ds+", -1.0)
        test_pairwise_accuracy.append(eval_test_dataset_result[0])
        test_one_of_many_accuracy.append(eval_test_dataset_result[1])
    else:
        eval_test_dataset_result = evaluate_test_dataset(args, model, device, dataset_name, -1.0)
        test_pairwise_accuracy.append(eval_test_dataset_result[0])
        test_one_of_many_accuracy.append(eval_test_dataset_result[1])

    result_dict['test_pairwise_accuracy'] = test_pairwise_accuracy
    result_dict['test_accuracy_one_vs_many'] = test_one_of_many_accuracy


def run_ablation(args, model, device, dataset_name, result_dict, optimizer):
    for dataset in ablation_dataset():
        data_name_temp = dataset.split("/")[4]
        update_arguments_for_dataset(data_name_temp, args)
        data_loader_evaluation_dataset = get_custom_dataset(args, dataset, True, True)
        accuracy_prediction = predict_batch(args, model, device, dataset,
                                            desc="Compute pairwise accuracy on ablation dataset {}".format(
                                                data_name_temp))
        result_dict[(data_name_temp + "-pairwise_accuracy")] = '{:.2f}%, {:.2f}%,[{}/{}]'.format(accuracy_prediction[0],
                                                                                                 accuracy_prediction[1],
                                                                                                 accuracy_prediction[2],
                                                                                                 accuracy_prediction[3])


def run_model(model_name, dataset_name, data_type, manual_type, data_mode_rnn, experiment_type, return_dict, gpu_id,
              seed,
              func, is_train, model_path, should_cache):
    args = get_arguments(model_name, dataset_name, data_type, manual_type, data_mode_rnn, experiment_type, seed,
                         model_path, should_cache)

    if is_train:
        args.save_model = True
        args.load_model = False

    use_cuda = args.use_gpu and torch.cuda.is_available()
    print("Cuda available: ", torch.cuda.is_available())
    torch.manual_seed(args.seed)

    if use_cuda:
        device_name = "cuda:" + str(gpu_id)
        # print("Cuda count ", torch.cuda.device_count())
        # print ('Available devices ', torch.cuda.device_count())
        print('Selected device: ' + device_name)
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")

    # since we have ~ 16 negative samples for 1 positive one during training
    args.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 16]).to(device))
    model = model_for_name(args, device)
    args.device = device
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if (args.load_model or experiment_type == "pretraining1") and os.path.isfile(args.save_path):
        print("Loading model", args.save_path, flush=True)
        checkpoint = torch.load(args.save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.epoch_start = checkpoint['epoch']
        # print("Model: {}, dataset {}, epoch {}".format(modelName, datasetName, checkpoint['epoch']))
    else:
        print("No model preloaded", args.save_path)

    start = time.time()
    result_dict = {}

    func(args, model, device, dataset_name, result_dict, optimizer)

    end = time.time()
    result_dict['execution_time'] = round((end - start), 3)
    return_dict_key = model_name + "-" + args.datasetName + "-" + data_to_string(
        args.data_type) + "-" + manual_type_to_string(args.manualDataType) + "-" + data_mode_rnn_to_string(
        args.manualFeatureType)
    return_dict[return_dict_key] = result_dict

    if args.save_model:
        torch.save({'epoch': args.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, args.save_path)
    return -1
