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

from PIL import Image, ImageDraw
import os
from shutil import copyfile
from core.view import View
import numpy as np
import copy
import pprint
import os
import shutil
from tqdm import tqdm
from core.features.handcrafted_feature_functions import compute_centered_vertically_different_views, \
    compute_centered_horizontally_different_views, popular_margin_vertical, popular_margin_horizontal, \
    popular_aspect_ratio, compute_intersections, inside_screen, compute_similar_alignment_horizontally, \
    compute_similar_alignment_vertically, add_raw_coordinates, compute_centered_horizontally, \
    compute_centered_vertically, compute_same_dimensions_score

# categorize mistakes for evaluation

maxNumberOfCandidates = 17
device_width = 360  # 1440
device_height = 512  # 2560
directory = "./dataset/data/dsplus/test/"
target_directory = "./dataset/data/ablation_dataset/"
downsample = 4
prefix = "dsplus_"


def draw_views(views, device_width, device_height, target_name):
    image = Image.new('RGB', (int(device_width), int(device_height)))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 0), (device_width + 1, device_height + 1)), fill="white")
    for view in views:
        view.draw_downsampled(draw, downsample)
    try:
        image.save(target_name, "PNG")
    except OSError as e:
        print("Could not save image: ", target_name, e)


def read_views(path):
    views = []
    with open(path, "r") as ins:
        for line in ins:
            line = line.replace(" ", "").replace("\n", "")
            numbers = line.split(",")
            views.append(View(int(int(numbers[0])), int(int(numbers[1])), int(int(numbers[2])), int(int(numbers[3]))))
    if len(views) == 0:
        print(path)
    return views


def create_directory_if_necessary(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def transfer_files(good_views, bad_views, original_views, good_filename, bad_filename, original_file_name, directory,
                   target):
    create_directory_if_necessary(target)
    copyfile(directory + good_filename, target + good_filename)
    draw_views(good_views, device_width, device_height, target + good_filename.split(".txt")[0] + ".png")

    copyfile(directory + bad_filename, target + bad_filename)
    draw_views(bad_views, device_width, device_height, target + bad_filename.split(".txt")[0] + ".png")

    copyfile(directory + original_file_name, target + original_file_name)
    draw_views(original_views, device_width, device_height, target + original_file_name.split(".txt")[0] + ".png")


# check that there are not more than 1
def differing_view(views, bad_views):
    # watch out for non rico datasets!
    for i, val in enumerate(views):
        if not views[i].equal(bad_views[i]):
            return i
    return -1


# watch out when changing the order in compute_vector to adapt the indexes...
def naming_map():
    return {"perserve_inside_screeen": [0, 0],
            "perserve_intersections": [1, 1],
            "perserve_margin_0_horizontally": [2, 2],
            "perserve_margin_horizontally": [3, 10],
            "perserve_margin_0_vertically": [11, 11],
            "perserve_margin_vertically": [12, 19],
            "perserve_aspect_ratio1-0": [20, 20],
            "perserve_centering_horizontally_one_view": [21, 21],
            "perserve_centering_horizontally_views": [22, 22],
            "perserve_centering_vertically_one_view": [23, 23],
            "perserve_centering_vertically_views": [24, 24],
            "perserve_similar_dimensions": [25, 25],
            "perserve_popular_aspect_ratios": [26, 26],
            }


def compute_handcrafted_vector(views):
    vector = []
    vector.append(inside_screen(views, views[0].width, views[0].height))
    vector.append(compute_intersections(views))
    vector.append(compute_similar_alignment_horizontally(views))
    for i in [8, 14, 16, 20, 24, 30, 32, 48]:
        vector.append(popular_margin_horizontal(views, [i * 2]))
    vector.append(compute_similar_alignment_vertically(views))
    for i in [8, 14, 16, 20, 24, 30, 32, 48]:
        vector.append(popular_margin_vertical(views, [i * 2]))
    vector.append(popular_aspect_ratio(views, [1.0 / 1.0]))
    vector.append(compute_centered_horizontally(views))
    vector.append(compute_centered_horizontally_different_views(views))
    vector.append(compute_centered_vertically(views))
    vector.append(compute_centered_vertically_different_views(views))
    vector.append(compute_same_dimensions_score(views))
    vector.append(
        popular_aspect_ratio(views, [9.0 / 16.0, 9.0 / 16.0]) + popular_aspect_ratio(views, [3.0 / 4.0, 4.0 / 3.0]))
    return vector


def compute_vector(views, views_original):
    vector = []
    array1 = compute_handcrafted_vector(views)
    array_org1 = compute_handcrafted_vector(views_original)
    vector = (np.asarray(array1) - np.asarray(array_org1)).tolist()
    return vector


mistakes = np.zeros(27)


def good_file(bad_name, root_dir):
    # 16 candidates
    for i in range(0, maxNumberOfCandidates):
        name = bad_name.split("-")[0] + "-" + bad_name.split("-")[1] + "-" + bad_name.split("-")[2] + "-" + str(
            i) + "_1.txt"
        if os.path.isfile(os.path.join(root_dir, name)):
            return True, name
    name = bad_name.split("-")[0] + "-" + bad_name.split("-")[1] + "-" + bad_name.split("-")[2] + "-tr_1.txt"
    if os.path.isfile(os.path.join(root_dir, name)):
        return True, name

    print("Good file does not exist for ", bad_name)
    return False, "Does not exist"


def original_file(filename):
    return filename.split("-")[0] + "-" + filename.split("-")[2] + "-original.txt"


# check which features appear with each other
correlations = {}
for key in naming_map().keys():
    correlations[key] = {}
    for key1 in naming_map().keys():
        correlations[key][key1] = 0

yes = {'yes', 'y', 'ye', ''}
no = {'no', 'n'}

if os.path.isdir(target_directory):
    print("Folder already exists on,", target_directory)
    choice = input("Do you want to delete the existing folder? ").lower()
    if choice in yes:
        print("Deleting existing folder")
        shutil.rmtree(target_directory)
    elif choice in no:
        print("Aborting")
        exit()
    else:
        sys.stdout.write("Please respond with 'yes' or 'no'")

print("Creating folder on", target_directory)

fileList = [s for s in os.listdir(directory) if
            ("_0.txt" in s)]  # and (sum(1 for line in open(os.path.join(directory,s))) == i))]
numberOfUniqueSamples = 0
for k, bad_filename in enumerate(tqdm(fileList)):
    bad_views = read_views(directory + bad_filename)

    good_filename = good_file(bad_filename, directory)[1]
    good_views = read_views(directory + good_filename)

    original_file_name = original_file(bad_filename)
    original_views = read_views(directory + original_file_name)
    if os.path.isfile(directory + good_filename):
        full = np.asarray(compute_vector(bad_views, original_views))
        with_distn = np.asarray(compute_vector(good_views, original_views))

        res = (full - with_distn)
        res = abs(res)
        mistakes = mistakes + res
        categories = []
        for key, indexes in naming_map().items():
            # print(key)
            # print(res[indexes[0]:indexes[1]+1])
            # +1 since it is excluding the upper limit

            # non exclusive property
            if res[indexes[0]:indexes[1] + 1].sum() != 0:
                # if we want the exclusive property: -> not a single one is true there
                if res.sum == res[indexes[0]:indexes[1] + 1].sum():
                    numberOfUniqueSamples = numberOfUniqueSamples + 1
                target = target_directory + "/" + prefix + key + "/"  # _directory + "/" + key + "/"
                transfer_files(good_views, bad_views, original_views, good_filename, bad_filename, original_file_name,
                               directory, target)
                categories.append(key)

        for category in categories:
            for tcategory in categories:
                correlations[category][tcategory] = correlations[category][tcategory] + 1

# print(fileList)
np.set_printoptions(suppress=True)
# print(mistakes)
correlationsVerbose = copy.deepcopy(correlations)
for category in correlations.keys():
    print("category", category)
    for tcategory in correlations.keys():
        percentage = -1
        if float(correlations[category][category]) > 0:
            percentage = float(correlations[category][tcategory]) / float(correlations[category][category])
        correlationsVerbose[category][tcategory] = '{}, {:.2f}%'.format(correlations[category][tcategory], percentage)

pp = pprint.PrettyPrinter(depth=6)
pp.pprint(correlationsVerbose)
print("numberOfUniqueSamples", numberOfUniqueSamples)
