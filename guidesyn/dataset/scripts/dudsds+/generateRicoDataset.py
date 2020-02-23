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
import json
from pprint import pprint
from View import View
from stack import Stack
from random import randint, uniform
import config
from helper import draw_subviews, draw_views, views_string
import os
import shutil
from random import choices
import copy
import numpy as np
import random
from tqdm import tqdm

# Data is downloaded from: http://rico.interactionmining.org
# 20 000 apps
# 1888471 views -> 100 views on average

# stats
numberOfViewsSmaller10 = 0
view_counter = 0  # how many views are rendered in total
error_counter = 0  # how often view is generated outside of the screen
theoreticalOutsideCounter = 0  # if bounds would be considered x, y, width, height


def extract_views(_id, rico_dir):
    path = os.path.join(rico_dir, 'combined', str(_id) + ".json")
    if not os.path.exists(path):
        return None
    #rico dataset location (downloaded from http://rico.interactionmining.org)
    with open(path) as f:
        data = json.load(f)
        data = data['activity']
        data = data['root']
        stack = Stack()
        stack.push(data)
        views = []

        while not stack.is_empty():
            top = stack.peek()
            stack.pop()
            bounds = top['bounds']
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            if width <= 0 or height <= 0:
                global error_counter
                error_counter = error_counter + 1

            subchildren = top.get('children')
            if subchildren is not None:
                for child in subchildren:
                    if child is None:
                        continue
                    stack.push(child)

            # Filter out really small views
            if width <= 6 or height <= 6:
                global numberOfViewsSmaller10
                numberOfViewsSmaller10 = numberOfViewsSmaller10 + 1
                continue

            views.append(View(0, bounds[0], bounds[1], width, height, 0, 0))
            global view_counter
            view_counter += len(views)
        return views


def move_view_randomly(views, rand_int, adapt_mode):
    if adapt_mode == 0:  # change horizontal: change view size
        views_size_mod = 72  # 1/20 of 1440
        old_view_size = views[rand_int].width
        views[rand_int].width = min(randint(old_view_size - views_size_mod, old_view_size + (views_size_mod / 2)),
                                    config.screen_width)
        views[rand_int].x += int((old_view_size - views[rand_int].width) / 2)
    elif adapt_mode == 1:  # change vertical: change view size
        views_size_mod_v = 128  # 1/20 of 1440
        old_view_size_height = views[rand_int].height
        views[rand_int].height = min(
            randint(old_view_size_height - views_size_mod_v, old_view_size_height + (views_size_mod_v / 2)),
            config.screen_height)
        views[rand_int].y += int((old_view_size_height - views[rand_int].height) / 2)
    elif adapt_mode == 2:  # change horizontal: change position
        views[rand_int].x += randint(-72, 72)  # randint(-18, 18)*4
    elif adapt_mode == 3:  # change vertical: change position
        views[rand_int].y += randint(-128, 128)  # randint(-32, 32)*4


# returns the id of the modified view
def adapt_view_randomly(views, max_views, view=None):
    # don't change content frame
    rand_int = randint(1, min(len(views) - 1, max_views - 1))
    original_view = views[rand_int].copy()

    # Multiple attempts (10) to avoid getting a view with exactly the same coordindates
    for i in range(0, 10):
        # either adapt horizontal or vertical constraint
        adapt_mode = randint(0, 3)
        # print("Adapt mode ", adaptMode)
        move_view_randomly(views, rand_int, adapt_mode)
        if not original_view.equal_size(views[rand_int]):
            break
    return rand_int


def string_to_arr(string):
    splitted = string.split("|")
    result = []
    for i in range(1, len(splitted)):
        result.append(int(splitted[i]))
    return result


# extracted from data_analyser.py in ds+
trans = [('|40|0|-40|0', 2975), ('|0|0|-40|0', 2958), ('|0|20|0|0', 2465), ('|0|-20|0|0', 1742), ('|0|40|0|0', 1604),
         ('|40|20|-40|0', 1345), ('|20|0|0|0', 1342), ('|20|0|-40|0', 1328), ('|0|20|-40|0', 1324),
         ('|-20|0|0|0', 1155), ('|0|0|0|40', 1150), ('|40|40|-40|0', 1014), ('|40|-20|-40|0', 1004),
         ('|0|40|-40|0', 1001), ('|20|20|0|0', 965), ('|0|-20|-40|0', 956), ('|0|0|0|-40', 914), ('|0|40|0|-40', 911),
         ('|0|-40|0|0', 896), ('|-20|0|40|0', 885), ('|40|0|-40|-40', 885), ('|0|0|-40|-40', 884),
         ('|40|40|-40|-40', 883), ('|0|40|-40|-40', 883), ('|0|0|-40|40', 863), ('|40|0|-40|40', 863),
         ('|-20|20|0|0', 825), ('|20|20|-40|0', 764), ('|0|-40|0|40', 730), ('|20|-20|-40|0', 684),
         ('|0|-40|-40|0', 664), ('|40|-40|-40|0', 663), ('|20|-20|0|0', 659), ('|0|10|0|0', 614),
         ('|0|-40|-40|40', 601), ('|-20|20|40|0', 590), ('|40|-40|-40|40', 584), ('|0|20|0|-40', 532),
         ('|-20|-20|0|0', 529), ('|0|-10|0|0', 515), ('|40|20|-40|-40', 505), ('|0|20|-40|-40', 504),
         ('|20|40|-40|0', 491), ('|20|-40|-40|0', 429), ('|20|40|0|0', 428), ('|20|0|-40|40', 383),
         ('|-20|-20|40|0', 382), ('|10|0|0|0', 367), ('|20|-40|-40|40', 365), ('|0|-20|0|40', 357),
         ('|-20|40|0|0', 318), ('|20|10|0|0', 283), ('|40|-10|-40|0', 260), ('|-10|0|0|0', 254), ('|40|10|-40|0', 231),
         ('|-20|40|40|0', 230), ('|40|0|0|0', 226), ('|20|0|0|40', 215), ('|0|10|-40|0', 207), ('|0|-10|-40|0', 206),
         ('|10|20|0|0', 197), ('|20|-40|0|0', 191), ('|-20|-40|0|0', 186), ('|20|-10|0|0', 181), ('|-20|10|0|0', 180),
         ('|40|40|0|0', 173), ('|-20|0|0|40', 168), ('|20|-10|-40|0', 165), ('|40|20|0|0', 161), ('|0|0|40|0', 155),
         ('|20|-20|0|40', 154), ('|40|-20|-40|40', 146), ('|-20|-40|40|0', 143), ('|-20|0|40|40', 139),
         ('|-40|0|0|0', 136), ('|-20|0|20|0', 133), ('|0|0|20|0', 129), ('|0|-20|-40|40', 129), ('|0|20|40|0', 126),
         ('|20|10|-40|0', 125), ('|0|40|40|0', 123), ('|-20|-20|0|40', 120), ('|-10|20|0|0', 118),
         ('|-20|-20|40|40', 114), ('|-20|-10|0|0', 110), ('|0|0|0|20', 103), ('|20|-40|0|40', 102), ('|0|-20|0|20', 98),
         ('|-20|-40|0|40', 94), ('|-40|0|40|0', 93), ('|-40|20|0|0', 92), ('|20|-20|-40|40', 87), ('|-40|40|0|0', 81),
         ('|10|-20|0|0', 77), ('|-20|-40|40|40', 75), ('|20|0|-40|-40', 75), ('|10|40|0|0', 74), ('|20|40|-40|-40', 73),
         ('|-20|-10|40|0', 71), ('|-20|10|40|0', 70), ('|40|0|0|40', 70), ('|20|20|-40|-40', 66), ('|-40|20|40|0', 66),
         ('|0|0|40|40', 56), ('|-40|40|40|0', 51), ('|0|-30|0|0', 49), ('|10|-40|0|0', 49), ('|-20|20|20|0', 43),
         ('|-40|-20|0|0', 41), ('|-40|0|0|40', 39), ('|-10|-20|0|0', 38), ('|10|10|0|0', 37), ('|-10|40|0|0', 37),
         ('|-40|-20|40|0', 36), ('|0|20|20|0', 34), ('|40|10|0|0', 34), ('|-40|10|0|0', 34), ('|-40|0|40|40', 32),
         ('|20|0|0|-40', 29), ('|20|40|0|-40', 29), ('|40|-20|0|0', 28), ('|-20|0|0|-40', 26), ('|-20|40|40|-40', 26),
         ('|20|20|0|-40', 26), ('|-20|20|0|-40', 25), ('|-20|40|0|-40', 25), ('|-40|10|40|0', 24), ('|0|10|40|0', 24),
         ('|-40|-40|0|0', 23), ('|30|0|0|0', 23), ('|20|-20|0|20', 23), ('|0|0|-40|20', 21), ('|-20|0|40|-40', 21),
         ('|40|-20|-40|20', 20), ('|-10|10|0|0', 20), ('|-10|-40|0|0', 20), ('|-20|20|40|-40', 18),
         ('|-40|-40|40|0', 18), ('|0|-20|40|0', 18), ('|20|-20|-40|20', 16), ('|40|-40|0|0', 16), ('|0|30|0|0', 15),
         ('|-40|-40|0|40', 14), ('|0|-30|-40|0', 14), ('|-40|-40|40|40', 13), ('|0|-40|40|0', 13), ('|0|-40|20|0', 10),
         ('|30|40|0|0', 10), ('|-20|-20|20|0', 10), ('|0|-20|-40|20', 9), ('|40|30|-40|0', 8), ('|-10|40|0|-40', 8),
         ('|20|0|-20|0', 8), ('|40|-10|0|0', 7), ('|0|40|20|0', 7), ('|0|0|0|-20', 7), ('|0|20|0|-20', 7),
         ('|40|0|-40|20', 7), ('|40|-40|0|40', 7), ('|-20|-20|0|20', 7), ('|0|30|-40|0', 7), ('|0|5|0|0', 6),
         ('|30|20|0|0', 6), ('|40|0|-40|-20', 6), ('|0|0|-40|-20', 6), ('|0|20|-40|-20', 6), ('|40|20|-40|-20', 6),
         ('|-40|-10|0|0', 6), ('|10|40|0|-40', 5), ('|-20|-40|20|0', 5), ('|0|10|0|-20', 5), ('|20|0|0|20', 5),
         ('|20|30|0|0', 5), ('|-10|-40|0|40', 5), ('|0|40|20|-40', 4), ('|0|10|-40|-20', 4), ('|40|10|-40|-20', 4),
         ('|0|-20|0|-20', 4), ('|20|0|-40|20', 4), ('|0|-40|40|40', 4), ('|10|-10|0|0', 4), ('|-20|40|20|0', 4),
         ('|20|-20|-20|0', 4), ('|10|-40|0|40', 4), ('|10|0|0|40', 3), ('|20|5|0|0', 3), ('|40|-20|-40|-20', 3),
         ('|0|-20|-40|-20', 3), ('|-40|-10|40|0', 3), ('|0|-5|0|0', 3), ('|20|-5|0|0', 3), ('|0|-10|0|40', 3),
         ('|-20|30|0|0', 3), ('|-30|0|0|0', 3), ('|0|40|0|-20', 3), ('|40|40|-40|-20', 3), ('|0|40|-40|-20', 3),
         ('|30|0|-40|0', 3), ('|-20|-10|20|0', 3), ('|20|-40|-20|0', 3), ('|-10|0|0|-40', 3), ('|-40|30|0|0', 3),
         ('|-10|0|0|40', 3), ('|30|-20|0|0', 3), ('|20|30|-40|0', 2), ('|0|0|20|-40', 2), ('|0|5|-40|0', 2),
         ('|-30|40|0|0', 2), ('|-30|20|0|0', 2), ('|0|-10|0|20', 2), ('|0|-10|-40|20', 2), ('|0|-10|-40|40', 2),
         ('|40|40|0|-40', 2), ('|40|0|0|-40', 2), ('|0|0|40|-40', 2), ('|0|40|40|-40', 2), ('|20|-30|0|0', 2),
         ('|10|0|-40|0', 2), ('|10|20|-40|0', 2), ('|20|-20|-20|20', 2), ('|20|-40|-20|40', 2), ('|20|-10|-20|0', 2),
         ('|-20|0|-20|0', 2), ('|10|-20|0|40', 2), ('|0|0|-20|0', 2), ('|30|20|-40|0', 2), ('|30|40|-40|0', 2),
         ('|0|-20|20|0', 2), ('|-10|-10|0|0', 2), ('|30|-40|0|0', 2), ('|-20|40|20|-40', 2), ('|0|0|20|40', 1),
         ('|40|5|-40|0', 1), ('|0|0|0|10', 1), ('|-20|0|20|-40', 1), ('|40|0|0|20', 1), ('|0|15|0|0', 1),
         ('|0|15|-40|0', 1), ('|-20|-5|0|0', 1), ('|20|-10|0|40', 1), ('|-20|30|40|0', 1), ('|-20|-10|0|40', 1),
         ('|-20|-10|40|40', 1), ('|10|0|0|-40', 1), ('|20|-30|-40|0', 1), ('|-30|-40|0|0', 1), ('|-30|-20|0|0', 1),
         ('|-30|-10|0|0', 1), ('|10|40|-40|0', 1), ('|-20|-40|20|40', 1), ('|-20|-40|-20|40', 1), ('|-20|-10|-20|0', 1),
         ('|-20|-20|-20|0', 1), ('|-20|-40|-20|0', 1), ('|0|10|20|0', 1), ('|-10|-20|0|40', 1), ('|-40|0|0|20', 1),
         ('|10|0|-20|0', 1), ('|20|40|-20|0', 1), ('|0|40|-20|0', 1), ('|10|40|-20|0', 1), ('|0|-10|20|0', 1),
         ('|-20|20|0|-20', 1), ('|-20|0|0|-20', 1), ('|20|10|0|-20', 1), ('|20|20|0|-20', 1), ('|20|0|0|-20', 1),
         ('|-20|10|40|-20', 1), ('|20|-20|0|-20', 1), ('|-20|0|40|-20', 1), ('|-20|-20|40|-20', 1),
         ('|-20|10|0|-20', 1), ('|-20|20|40|-20', 1), ('|-20|-20|0|-20', 1), ('|-20|-30|0|0', 1), ('|-20|-30|40|0', 1),
         ('|0|20|40|-40', 1), ('|40|20|0|-40', 1), ('|-20|-20|20|40', 1), ('|40|30|0|0', 1)]
trans_sum = sum([s[1] for s in trans])
transformation_weights = [float(s[1]) / float(trans_sum) for s in trans]
transformations = [string_to_arr(s[0]) for s in trans]


def adapt_view_according_to_synthesis(views, transformation_id, mod_view):
    transformation = transformations[transformation_id]
    view = views[mod_view]
    view.x = view.x - transformation[0]
    view.y = view.y - transformation[1]
    view.width = view.width - transformation[2]
    view.height = view.height - transformation[3]


def modify_views(views, app_id, folder, max_views):
    # config
    number_of_candidates = randint(config.number_of_candidates_lower, config.number_of_candidates_upper)
    modifications = []
    trans_ids = []

    pos_views = copy.deepcopy(views)

    # Experiment: more robust to changes in earlier layers
    if len(views) >= 3 and config.modify_more_than_last_view:
        number_of_modifications = random.randint(0, min(config.number_of_view_modifications_upper, len(views) - 2))
        # range is < upper
        modifications = random.sample(range(1, len(views) - 1), number_of_modifications)
        trans_ids = np.random.choice(len(transformation_weights), size=number_of_modifications, replace=False,
                                     p=transformation_weights)
    for m, mod in enumerate(modifications):
        adapt_view_according_to_synthesis(pos_views, trans_ids[m], mod)

    image_name = folder + str(app_id) + "_1.png"
    # draw_views(views_to_be_drawn, image_name)
    with open((folder + str(app_id) + "_1.txt"), "w+") as text_file:
        text_file.write(views_string(pos_views))

    du_folder = "./" + folder.split("/")[1] + "-du/" + folder.split("/")[2] + "/"
    with open((du_folder + str(app_id) + "_1.txt"), "w+") as text_file:
        text_file.write(views_string(pos_views))

    transformation_ids = np.random.choice(len(transformation_weights), size=number_of_candidates, replace=False,
                                          p=transformation_weights)

    for j in range(0, number_of_candidates):
        false_views = copy.deepcopy(pos_views)
        adapt_view_according_to_synthesis(false_views, transformation_ids[j], len(views) - 1)
        # draw_views(views_to_be_drawn, image_name)
        with open((folder + str(app_id) + "_" + str(j) + "_0.txt"), "w+") as text_file:
            text_file.write(views_string(false_views))


# remove duplicates, filter out same views
def filter_duplicates(views):
    new_views = []
    for view in views:
        to_add = True
        for view1 in new_views:
            if view.equal_coordinates(view1):  # do not add
                # print("Filter out")
                to_add = False
                break
        if to_add:
            new_views.append(view)
    return new_views


def preprocess_views(views, max_views):
    # sort by size
    views_sorted = sorted(views, key=lambda x: x.area(), reverse=True)
    # how many views we want in this iteration
    rand_int = randint(2, min(max_views, len(views_sorted)))

    views_to_be_drawn = []
    views_to_be_drawn.append(View(-999, 0, 0, config.screen_width, config.screen_height))
    for view in views_sorted:
        if view.width == config.screen_width and view.height == config.screen_height:
            continue
        if len(views_to_be_drawn) >= rand_int:
            break
        views_to_be_drawn.append(view)
    return views_to_be_drawn


# generates at max numberOfSamples: (since only views are generated with at upper_limit_views)
def generate_ds_du(upper_limit_views, number_of_samples, ds_dir, rico_dir):
    generated = 0
    if os.path.isdir(ds_dir):
        shutil.rmtree(ds_dir)
        shutil.rmtree(ds_dir + "-du")
    folder_name = ["/validate/", "/train/", "/test/"]
    for name in folder_name:
        os.makedirs(ds_dir + name)
        os.makedirs(ds_dir + "-du" + name)

    for i in tqdm(range(1, number_of_samples), desc="Generating samples", ncols=120):
        # for i in range(1, config.numberOfSamples):
        if i % config.eval_rate == 0:
            folder = ds_dir + "/validate/"
        elif i % config.eval_rate == 1:
            folder = ds_dir + "/test/"
        else:
            folder = ds_dir + "/train/"

        views = extract_views(i, rico_dir)
        if views is None:
            continue
        views = filter_duplicates(views)
        if len(views) <= 2:  # and len(views) < upper_limit_views):
            continue
        views = preprocess_views(views, upper_limit_views)  # try and except

        # image_name = folder + str(i) + "_original" + ".png" #str(0)
        # draw_views(views, image_name)
        modify_views(views, i, folder, upper_limit_views)
        generated = generated + 1
    return generated
