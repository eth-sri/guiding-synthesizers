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

import pprint
import argparse
from guidesyn.core.run_model import run_model, evaluate
from guidesyn.core.arguments import Data, ManualType, DataModeRNN
import time


def main():
    evaluations = [
        ("MLP", "ds", Data.MANUAL, ManualType.Manual_MLP_OS, DataModeRNN.UNUSED),
        ("CNN", "ds", Data.IMAGE, ManualType.UNUSED, DataModeRNN.UNUSED),
        ("RNN", "ds", Data.MANUAL, ManualType.Manual_RNN_OS, DataModeRNN.FULL),
        ("EnsembleRnnCnn", "ds", Data.BOTH, ManualType.Manual_RNN_OS, DataModeRNN.FULL),
        ("MLP", "dsplus", Data.MANUAL, ManualType.Manual_MLP_MS, DataModeRNN.UNUSED),
        ("CNN", "dsplus", Data.IMAGE, ManualType.UNUSED, DataModeRNN.UNUSED),
        ("RNN", "dsplus", Data.MANUAL, ManualType.Manual_RNN_MS, DataModeRNN.FULL),
        ("EnsembleRnnCnn", "dsplus", Data.BOTH, ManualType.Manual_RNN_MS, DataModeRNN.FULL)
    ]

    parser = argparse.ArgumentParser(description='Experiment specifications')
    parser.add_argument('--gpus', type=str, default="-1", help='GPU to be used')
    parser.add_argument('--model_path', type=str, default="./core/saved_modelsPaper/",
                        help='Path to load models from/save to')
    parser.add_argument('--should_cache', type=bool, default="False", help='Should cache neural net input.')
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    return_dict = {}

    start = time.time()
    print("Oracle evaluations for {} models ".format(len(evaluations)))
    for i, (model, dataset, dataType, manualType, DataModeRnn) in enumerate(evaluations):
        print("{}.) Evaluation: {} trained on {}".format(i, model, dataset), flush=True)
        run_model(model, dataset, dataType, manualType, DataModeRnn, "models-acc", return_dict, gpus[0], 42, evaluate,
                  False, args.model_path, args.should_cache)
        print("\n----------------------------------------")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(return_dict)
    end = time.time()
    print("Running evaluation in {}s".format(end - start))


if __name__ == '__main__':
    main()
