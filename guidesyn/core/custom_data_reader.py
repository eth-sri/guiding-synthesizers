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

from __future__ import print_function
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from skimage import io, transform
import os
from PIL import Image, ImageDraw
import copy
import numpy as np
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence
from joblib import Memory
import time
from guidesyn.core.view import View, Constraint
from guidesyn.core.features.handcrafted_feature_functions import compute_centered_vertically_different_views, \
    compute_centered_horizontally_different_views, popular_margin_vertical, popular_margin_horizontal, \
    popular_aspect_ratio, compute_intersections, inside_screen, compute_similar_alignment_horizontally, \
    compute_similar_alignment_vertically, add_raw_coordinates, compute_centered_horizontally, \
    compute_centered_vertically, compute_same_dimensions_score
from guidesyn.core.arguments import Data, ManualType, DataModeRNN
from guidesyn.core.features.RNN_feature_functions_multiple_screens import compute_horizontal_distance, \
    compute_vertical_distance, compute_dimension, compute_ratio
from guidesyn.core.features.RNN_feature_functions_one_screen import compute_horizontal_distance_one_screen, \
    compute_vertical_distance_one_screen, compute_dimension_one_screen, compute_ratio_one_screen
import argparse

cachedir = os.path.abspath(__file__).replace(".py", "") + '-cache/'
memory = Memory(cachedir, verbose=0, compress=True)


def floor_and_invert_transform(tensor):
    # We expect the data to be sparse
    # print(str(torch.nonzero(x.data).size(0)) + " / " + str(matrix_entries))
    # assert(torch.nonzero(x.data).size(0) <=  matrix_entries/10)
    # Invert the image, since we want a 1 if the pixel is black and 0 otherwise
    return 1.0 - tensor.floor()


def custom_key(view):
    return view.area(), view.x


def read_views(path, dx_to_dp=1):
    views = []
    with open(path, "r") as ins:
        for line in ins:
            line = line.replace(" ", "").replace("\n", "")
            numbers = line.split(",")
            if len(numbers) == 4:
                views.append(
                    View(int(int(numbers[0]) / dx_to_dp), int(int(numbers[1]) / dx_to_dp),
                         int(int(numbers[2]) / dx_to_dp),
                         int(int(numbers[3]) / dx_to_dp), 0.0, 0.0))
            else:
                view = View(int(int(numbers[0]) / dx_to_dp), int(int(numbers[1]) / dx_to_dp),
                            int(int(numbers[2]) / dx_to_dp),
                            int(int(numbers[3]) / dx_to_dp), float(numbers[4]), float(numbers[5]))
                if len(numbers) > 6:
                    vertical = Constraint(numbers[6], float(numbers[7]), numbers[8], int(numbers[9]), int(numbers[10]),
                                          float(numbers[11]), int(numbers[12]), int(numbers[13]), int(numbers[14]))
                    horizontal = Constraint(numbers[15], float(numbers[16]), numbers[17], int(numbers[18]),
                                            int(numbers[19]), float(numbers[20]), int(numbers[21]), int(numbers[22]),
                                            int(numbers[23]))
                    view.add_constraints(vertical, horizontal)
                views.append(view)
    if len(views) == 0:
        print("No views for file, ", path)
    return views


def handcrafted_feature_vector(views):
    array = []
    # add feature functions
    div1 = len(views)
    div2 = len(views) * len(views)
    div3 = len(views) * len(views) * len(views)
    array.append(compute_intersections(views) / div2)
    array.append(compute_similar_alignment_vertically(views) / div2)
    array.append(compute_similar_alignment_horizontally(views) / div2)
    array.append(inside_screen(views, views[0].width, views[0].height) / div1)
    array.append(compute_centered_horizontally(views) / div2)
    array.append(compute_centered_vertically(views) / div2)
    array.append(compute_same_dimensions_score(views) / div2)
    # multiply by two because of dimensions 1440 x 2560
    for i in [0, 8, 14, 16, 20, 24, 30, 32, 48]:
        array.append(popular_margin_horizontal(views, [i * 2]) / div2)
    for i in [0, 8, 14, 16, 20, 24, 30, 32, 48]:
        array.append(popular_margin_vertical(views, [i * 2]) / div2)

    array.append(popular_aspect_ratio(views, [1.0 / 1.0]) / div1)
    array.append(popular_aspect_ratio(views, [3.0 / 4.0, 4.0 / 3.0]) / div1)
    array.append(popular_aspect_ratio(views, [9.0 / 16.0, 9.0 / 16.0]) / div1)
    array.append(compute_centered_vertically_different_views(views) / div3)
    array.append(compute_centered_horizontally_different_views(views) / div3)
    return array


# for the MLP -> handcrafted features derived from InferUI
def transform_views_to_handcrafted_features_one_screen(views):
    return torch.FloatTensor(handcrafted_feature_vector(views))


def transform_views_to_handcrafted_features_multiple_screens(views, views_original):
    array = handcrafted_feature_vector(views)
    array_org = handcrafted_feature_vector(views_original)
    array_diff = (np.asarray(array) - np.asarray(array_org)).tolist()
    # other possibility: just return the diff: torch.FloatTensor(array_diff)
    return torch.FloatTensor(array + array_org + array_diff)


def sort_by_area(views, filename=""):
    resort = []
    for i in range(0, len(views)):
        views[i].sortId = i
    views.sort(key=custom_key, reverse=True)
    for i in range(0, len(views)):
        resort.append(views[i].sortId)
    if views[0].sortId != 0:
        pass
        # print(filename)
        # print("FATAL ERROR. This views content frame is switched")
    return resort


# for RNN for multiple screens
def transform_views_to_simple_features_multiple_screens(views, original_views, mode):
    views = copy.deepcopy(views)
    resort = sort_by_area(views)
    # sortByArea(views_original)
    views_original = []
    for i in range(0, len(resort)):
        views_original.append(original_views[resort[i]])

    for i in range(0, len(views) - 1):
        # prerequisite: arrays are sorted
        # print(views[i].area(), views[i].width, views[i].height)
        if views[i].area() < views[i + 1].area():
            exit(0)
            print("FATAL")

    array = []
    for i in range(0, len(views)):
        first_rnn_bundle = []
        for j in range(0, len(views)):
            if i != j:
                second_rnn_bundle = []
                second_rnn_bundle.extend(compute_horizontal_distance(views_original, views, i, j, mode))
                second_rnn_bundle.extend(compute_vertical_distance(views_original, views, i, j, mode))
                second_rnn_bundle.extend(compute_dimension(views_original, views, i, j, mode))
                second_rnn_bundle.extend(compute_ratio(views_original, views, i, j, mode))
                first_rnn_bundle.append(second_rnn_bundle)
        array.append(first_rnn_bundle)
    return torch.FloatTensor(array)


def transform_views_to_simple_features_one_screen(views, mode):
    views = copy.deepcopy(views)
    sort_by_area(views)

    for i in range(0, len(views) - 1):
        # prerequisite: arrays are sorted
        # print(views[i].area(), views[i].width, views[i].height)
        if views[i].area() < views[i + 1].area():
            exit(0)  # maybe uncomment
            print("FATAL")

    array = []
    # already_processed_views = []
    for i in range(0, len(views)):
        first_rnn_bundle = []
        for j in range(0, len(views)):
            if i != j:
                second_rnn_bundle = []
                second_rnn_bundle.extend(compute_horizontal_distance_one_screen(views, i, j, mode))
                second_rnn_bundle.extend(compute_vertical_distance_one_screen(views, i, j, mode))
                second_rnn_bundle.extend(compute_dimension_one_screen(views, i, j, mode))
                second_rnn_bundle.extend(compute_ratio_one_screen(views, i, j, mode))
                first_rnn_bundle.append(second_rnn_bundle)
        array.append(first_rnn_bundle)
    return torch.FloatTensor(array)


def sort_by_order(view_order, views):
    for i in range(0, len(views)):
        swap_index = view_order[i]
        temp = views[i]
        views[i] = views[swap_index]
        views[swap_index] = temp


def draw_image(args, views, is_eval, filename=""):
    width = args.target_image_width
    height = args.target_image_height
    downsample = args.downsample

    image = Image.new('RGB', (int(width), int(height)))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 0), (width + 1, height + 1)), fill="white")
    if is_eval:
        # center
        random_offset_x = (width - int((views[0].width / downsample))) / 2
        random_offset_y = (height - int((views[0].height / downsample))) / 2
    else:
        random_offset_x = random.randint(-int(views[0].x / downsample), width - int(views[0].x_end() / downsample))
        random_offset_y = random.randint(-int(views[0].y / downsample), height - int(views[0].y_end() / downsample))

    for view in views:
        view.draw_downsampled(draw, downsample, random_offset_x, random_offset_y)
    return image


# x = 0
def get_image_data(views, args, is_eval, filename=""):
    # global x
    # drawViews(views, "./debugEval/", str(x), args.target_image_width, args.target_image_height, True, args.downsample)
    # x = x + 1
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=args._in_channel),
                                    # transforms.Resize((args.image_height, args.image_width)),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: floor_and_invert_transform(x)),
                                    ])
    image = draw_image(args, views, is_eval, filename)
    # for visualisation, e.g. bad example
    # image.save(image_name, "PNG")
    # image = Image.open(path)
    image = transform(image.convert('RGB'))
    return image


def get_manual_data(args, views, original_views):
    manual = []
    if args.manualDataType == ManualType.Manual_RNN_MS:
        manual = transform_views_to_simple_features_multiple_screens(views, original_views, args.manualFeatureType)
    elif args.manualDataType == ManualType.Manual_RNN_OS:
        manual = transform_views_to_simple_features_one_screen(views, args.manualFeatureType)
    elif args.manualDataType == ManualType.Manual_MLP_MS:
        manual = transform_views_to_handcrafted_features_multiple_screens(views, original_views)
    elif args.manualDataType == ManualType.Manual_MLP_OS:
        manual = transform_views_to_handcrafted_features_one_screen(views)
    return manual


@memory.cache
def preprocess_file(filename, root_dir, args, is_eval):
    label_pos = len(filename.split('_')) - 1
    label = (int(filename.split('_')[label_pos].split('.')[0]))
    views_path = os.path.join(root_dir, (filename.split(".")[0] + ".txt"))

    if not os.path.isfile(views_path):
        print("Invalid views path: " + views_path)
        print(filename)
    image = []
    if args.data_type == Data.IMAGE or args.data_type == Data.BOTH:
        # Watch out here with downsample that it fits.
        # print(self.args.downsample)
        image = get_image_data(read_views(views_path), args, is_eval, filename)

    manual = []
    if args.data_type == Data.MANUAL or args.data_type == Data.BOTH or args.data_type == Data.IOP:
        if args.extraOriginal:
            original_file_name = filename.split("-")[0] + "-" + filename.split("-")[2] + "-original.txt"
        else:
            original_file_name = filename.split('_')[0] + "_1.txt"
        original_file_name_path = os.path.join(root_dir, original_file_name)
        views = read_views(views_path, args.pixToDp)
        original_views = read_views(original_file_name_path, args.pixToDp)
        manual = get_manual_data(args, views, original_views)
    sample = {'image': image, 'manual': manual, 'label': label}
    return sample


class CustomDataset(Dataset):

    def __init__(self, args, root_dir, file_list, is_eval):
        self.root_dir = root_dir
        if file_list is None:
            self.fileList = [x for x in os.listdir(self.root_dir) if (".txt" in x and "-original" not in x)]
            # to filter by number of views, e.g. only consider length 2
            # self.fileList = [s for s in temp if (sum(1 for line in open(os.path.join(self.root_dir,s))) == 4)]
        else:
            self.fileList = file_list
        self.args = args
        self.isEval = is_eval

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, idx):
        sample = preprocess_file(self.fileList[idx], self.root_dir, self.args.to_small(), self.isEval)
        return sample


def my_collate_rnn(batch):
    batch.sort(key=lambda x: x["manual"].size(), reverse=True)
    _list = []
    batch_counter = []
    for d in batch:
        dic = d["manual"]
        batch_counter.append(dic.size(0))
        for i in range(0, dic.size(0)):
            _list.append(dic[i, :, :])
    _pack_sequence = pack_sequence(_list)
    manual_data = pad_packed_sequence(_pack_sequence, batch_first=True)
    label = torch.tensor([(d["label"]) for d in batch])
    if len(batch[0]["image"]) > 0:
        image_list = [(d["image"]) for d in batch]
        images = torch.stack(image_list, 0)
        return {"manual": manual_data, "image": images, "label": label, "batch_counter": torch.tensor(batch_counter)}
    return {"manual": manual_data, "label": label, "batch_counter": torch.tensor(batch_counter)}


def get_custom_dataset(args, root_dir, shuffle, is_eval, file_list=None):
    if not args.cache:
        memory.location = None
    dataset = CustomDataset(args, root_dir, file_list, is_eval)
    if args.modelName == "RNN" or args.modelName == "EnsembleRnnCnn":
        data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=shuffle, num_workers=10, collate_fn=my_collate_rnn)
    else:
        data_loader = DataLoader(dataset, batch_size=args.batch_size,
                                 shuffle=shuffle, num_workers=10)
    return data_loader
