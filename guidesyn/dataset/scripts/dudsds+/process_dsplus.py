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

import os
import random
import shutil
import math
from distutils.dir_util import copy_tree
from generateRicoDataset import modify_views
from View import View
from tqdm import tqdm

foldername = ["/validate/", "/train/", "/test/"]


def read_views(path):
    views = []
    with open(path, "r") as ins:
        for line in ins:
            line = line.replace(" ", "").replace("\n", "")
            numbers = line.split(",")
            views.append(View(-999, int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])))
    if len(views) == 0:
        print(path)
    return views


# ds_path + "/test/" + filename, number of files + disjoint files
def compute_folder_stats(directory):
    files = {int(x.split("-")[0]) for x in os.listdir(directory) if "-original" in x}
    all_files = [x for x in os.listdir(directory)]
    print("Directory: ", directory)
    print("Disjoint apps: ", len(files), min(files), max(files))
    print("Total: ", len(all_files))
    print("Original: ", len([x for x in os.listdir(directory) if "-original" in x]))
    dict_smaller_device = {}
    dict_larger_device = {}

    for i, file in enumerate(all_files):
        if int(file.split("-")[1]) == 0:
            dict_smaller_device[int(file.split("-")[2])] = dict_smaller_device.get(int(file.split("-")[2]), 0) + 1

    print("dict_smaller_device")
    [print(key, " : ", value, end=', ') for (key, value) in sorted(dict_smaller_device.items())]
    print("-----")
    print("dict_larger_device")
    [print(key, " : ", value, end=', ') for (key, value) in sorted(dict_larger_device.items())]
    print("-----")


def split_test_validate_train(upper_limit_views, directory, percentage_test, percentage_eval, target_directory):
    if os.path.isdir(target_directory):
        shutil.rmtree(target_directory, ignore_errors=True)

    for name in foldername:
        os.makedirs(target_directory + name)

    files = {int(x.split("-")[0]) for x in os.listdir(directory) if "-original" in x}
    number_of_apps = len(files)
    print("Number of apps: ", number_of_apps)

    sorted_files = sorted(files)
    number_test = int(number_of_apps * percentage_test)
    number_eval = int(number_of_apps * percentage_eval)

    print("Test: ", sorted_files[0], sorted_files[number_test], number_test)
    print("Validate: ", sorted_files[number_test + 1], sorted_files[number_test + number_eval], number_eval)
    print("Train: ", sorted_files[number_test + number_eval + 1], sorted_files[number_of_apps - 1])
    # rest is in train

    upper_test = sorted_files[number_test]
    upper_eval = sorted_files[number_test + number_eval]
    copied = 0
    not_copied = 0

    file_list = [x for x in os.listdir(directory)]
    for i, filename in enumerate(file_list):
        if i % 500 == 0:
            print(i, len(file_list))
        app_id = int(filename.split("-")[0])
        if "original" in filename:
            should_copy = int(filename.split("-")[1]) <= upper_limit_views
        else:
            should_copy = int(filename.split("-")[2]) <= upper_limit_views

        if should_copy:
            copied = copied + 1
            if app_id <= upper_test:
                shutil.copyfile(directory + filename, target_directory + "/test/" + filename)
            elif app_id <= upper_eval:
                shutil.copyfile(directory + filename, target_directory + "/validate/" + filename)
            else:
                shutil.copyfile(directory + filename, target_directory + "/train/" + filename)
        else:
            not_copied = not_copied + 1
            print("Copied vs non copied", copied, not_copied)

    for name in foldername:
        compute_folder_stats(target_directory + name)
    return upper_test, upper_eval


def anaylse_folders(directory):
    for name in foldername:
        compute_folder_stats(directory + name)


def copy_dsplus_to_dsdu_folders(app_id, upper_limit, source_dir, target_dir):
    # copy over to d_s and d_u
    print("Copy to d_s and d_u")
    for f in foldername:
        print(f)
        file_list = [x for x in os.listdir(source_dir + f) if "original" in x]
        if not os.path.isdir(target_dir):
            print("Can not copy into folder which does not exist!")
        for i, filename in enumerate(tqdm(file_list, desc="Copying " + source_dir)):
            views = read_views(source_dir + f + filename)
            # transform to views and modify
            modify_views(views, app_id, target_dir + f, upper_limit)
            app_id = app_id + 1

