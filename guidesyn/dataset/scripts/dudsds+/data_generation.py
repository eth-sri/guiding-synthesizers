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

from generateRicoDataset import generate_ds_du

from process_dsplus import split_test_validate_train, copy_dsplus_to_dsdu_folders
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--upper_limit_views', nargs='?', const=30, type=int, default=30)
parser.add_argument('--number_of_samples_rico', nargs='?', const=20000, type=int, default=20000)
parser.add_argument('--dsplus_src_dir', nargs='?', const="./dsplusPlain", default="./dsplusPlain")
parser.add_argument('--dsplus_tar_dir', nargs='?', const="./dsplus", default="./dsplus")
parser.add_argument('--ds_dir', nargs='?', const="./ds", default="./ds")

args = parser.parse_args()
print(args.dsplus_src_dir)
print(args.number_of_samples_rico)
print(args.upper_limit_views)
print(args.dsplus_tar_dir)

generated = generate_ds_du(args.upper_limit_views, args.number_of_samples_rico, args.ds_dir)
print("Generated app samples: ", generated)
(upper_test, upper_validate) = split_test_validate_train(args.upper_limit_views, args.dsplus_src_dir, 0.18, 0.12,
                                                      args.dsplus_tar_dir)
# (upper_test, upper_validate) = (436, 694)
print("Spit into test, eval, train", upper_test, upper_validate)
copy_dsplus_to_dsdu_folders(generated + 1, args.upper_limit_views, args.dsplus_tar_dir, args.ds_dir)
