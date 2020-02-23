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
import argparse
import os

from generateRicoDataset import generate_ds_du

from process_dsplus import split_test_validate_train, copy_dsplus_to_dsdu_folders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upper_limit_views', nargs='?', const=30, type=int, default=30)
    parser.add_argument('--number_of_samples_rico', nargs='?', const=20000, type=int, default=20000)
    parser.add_argument('--dsplus_dir', default="./dataset/data/dsplus")
    parser.add_argument('--include_dsplus', default=False, action='store_true', help="Adds samples from DS+ dataset to DS to make them comparable")
    parser.add_argument('--ds_dir', nargs='?', const="./ds", default="./ds")
    parser.add_argument('--rico_dir', required=True, type=str, help="Path to the rico dataset")
    args = parser.parse_args()

    generated = generate_ds_du(args.upper_limit_views, args.number_of_samples_rico, args.ds_dir, args.rico_dir)

    if not args.include_dsplus:
        return
    if not os.path.exists(args.dsplus_dir):
        print('Unable to add samples from dataset DS+. Dataset not found in "{}"'.format(args.dsplus_dir))
        return
    copy_dsplus_to_dsdu_folders(generated + 1, args.upper_limit_views, args.dsplus_dir, args.ds_dir)


if __name__ == "__main__":
    main()
