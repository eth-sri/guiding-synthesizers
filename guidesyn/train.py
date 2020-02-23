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

from guidesyn.core.run_model import run_model, run_training
from multiprocessing import Process
import multiprocessing
import time
import pprint
from guidesyn.core.arguments import Data, ManualType, DataModeRNN
import os
import argparse
import pickle


def save_dictionary(dictionary, filename):
    with open(filename, "wb") as myFile:
        pickle.dump(dictionary, myFile)
        myFile.close()


def load_dictionary(filename):
    with open(filename, "rb") as myFile:
        dictionary = pickle.load(myFile)
        myFile.close()
        return dictionary


def get_training_config():
    return [("MLP", "dsplus", Data.MANUAL, ManualType.Manual_MLP_MS, DataModeRNN.UNUSED),
            ("MLP", "ds", Data.MANUAL, ManualType.Manual_MLP_OS, DataModeRNN.UNUSED),
            ("RNN", "dsplus", Data.MANUAL, ManualType.Manual_RNN_MS, DataModeRNN.FULL),
            ("RNN", "ds", Data.MANUAL, ManualType.Manual_RNN_OS, DataModeRNN.FULL),
            ("CNN", "dsplus", Data.IMAGE, ManualType.UNUSED, DataModeRNN.UNUSED),
            ("CNN", "ds", Data.IMAGE, ManualType.UNUSED, DataModeRNN.UNUSED),
            ("EnsembleRnnCnn", "ds", Data.BOTH, ManualType.Manual_RNN_OS, DataModeRNN.FULL),
            ("EnsembleRnnCnn", "dsplus", Data.BOTH, ManualType.Manual_RNN_MS, DataModeRNN.FULL)
            ]


def main():
    parser = argparse.ArgumentParser(description='Experiment specifications')
    parser.add_argument('--exp', type=str, default="models-acc", help='Type of exeriment')
    parser.add_argument('--gpus', type=str, default="-1", help='GPUs to be used')
    parser.add_argument('--seed', type=str, default="42", help='Random seed for pytorch')
    parser.add_argument('--model_path', type=str, default="./core/saved_modelsPaper/",
                        help='Path to load models from/save to')
    parser.add_argument('--should_cache', type=bool, default="False", help='Should cache neural net input.')
    args = parser.parse_args()
    experiment = args.exp
    gpus = args.gpus.split(",")
    seed = args.seed
    model_path = args.model_path

    if not os.path.isdir(model_path):
        print("Generating model save folder at path", model_path)
        os.makedirs(model_path)
    elif len(os.listdir(model_path)) != 0:
        choice = input("Folder {} already exists. Do you want to overwrite existing models?".format(model_path)).lower()
        if choice in {'no', 'n'}:
            print(
                "Aborting. Select and other folder path by passing an argument to --model_path or agree on overwriting excisting models.")
            exit()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    combinations = get_training_config()
    print(combinations)
    print("Experiments: ", len(combinations))

    for i, (model, dataset, dataType, manualType, DataModeRnn) in enumerate(combinations):
        p = Process(target=run_model, args=(
            model, dataset, dataType, manualType, DataModeRnn, experiment, return_dict, gpus[i % len(gpus)], seed,
            run_training, True, model_path, args.should_cache))
        processes.append(p)

    for i in range(min(len(gpus), len(processes))):
        time.sleep(5)
        processes[i].start()

    for i in range(len(processes)):
        processes[i].join()
        new_id = i + len(gpus)
        if new_id < len(processes):
            processes[new_id].start()

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(return_dict._getvalue())
    save_name = "./results/result_" + str(time.time()) + ".txt"
    save_dictionary(return_dict._getvalue(), save_name)


if __name__ == '__main__':
    main()
