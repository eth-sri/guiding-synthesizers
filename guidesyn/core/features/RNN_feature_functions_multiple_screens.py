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

from guidesyn.core.arguments import DataModeRNN


def compute_horizontal_distance(views_original, views, index, index_comp, mode):
    features = []
    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(views_original[index].x - views_original[index_comp].x)
        features.append(views[index].x - views[index_comp].x)
        features.append(views_original[index].x - views_original[index_comp].x_end())
        features.append(views[index].x - views[index_comp].x_end())
        features.append(views_original[index].x_end() - views_original[index_comp].x_end())
        features.append(views[index].x_end() - views[index_comp].x_end())
        features.append(views_original[index].x_end() - views_original[index_comp].x)
        features.append(views[index].x_end() - views[index_comp].x)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        features.append(
            (views_original[index].x - views_original[index_comp].x) - (views[index].x - views[index_comp].x))
        features.append(
            (views_original[index].x - views_original[index_comp].x_end()) - (
                        views[index].x - views[index_comp].x_end()))
        features.append((views_original[index].x_end() - views_original[index_comp].x_end()) - (
                views[index].x_end() - views[index_comp].x_end()))
        features.append(
            (views_original[index].x_end() - views_original[index_comp].x) - (
                        views[index].x_end() - views[index_comp].x))
    return features


def compute_vertical_distance(views_original, views, index, index_comp, mode):
    features = []
    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(views_original[index].y - views_original[index_comp].y)
        features.append(views[index].y - views[index_comp].y)
        features.append(views_original[index].y - views_original[index_comp].y_end())
        features.append(views[index].y - views[index_comp].y_end())
        features.append(views_original[index].y_end() - views_original[index_comp].y_end())
        features.append(views[index].y_end() - views[index_comp].y_end())
        features.append(views_original[index].y_end() - views_original[index_comp].y)
        features.append(views[index].y_end() - views[index_comp].y)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        features.append(
            (views_original[index].y - views_original[index_comp].y) - (views[index].y - views[index_comp].y))
        features.append(
            (views_original[index].y - views_original[index_comp].y_end()) - (
                        views[index].y - views[index_comp].y_end()))
        features.append((views_original[index].y_end() - views_original[index_comp].y_end()) - (
                views[index].y_end() - views[index_comp].y_end()))
        features.append(
            (views_original[index].y_end() - views_original[index_comp].y) - (
                        views[index].y_end() - views[index_comp].y))
    return features


# resizing
def compute_dimension(views_original, views, index, index_comp, mode):
    features = []
    h_diff_original = views_original[index].height - views_original[index_comp].height
    h_diff = views[index].height - views[index_comp].height
    w_diff_original = views_original[index].width - views_original[index_comp].width
    w_diff = views[index].width - views[index_comp].width

    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(h_diff)
        features.append(h_diff_original)
        features.append(w_diff)
        features.append(w_diff_original)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        features.append(h_diff_original - h_diff)
        features.append(w_diff_original - w_diff)
        features.append(h_diff_original + w_diff_original - (w_diff + h_diff))
    return features


def compute_ratio(views_original, views, index, index_comp, mode):
    features = []
    ratio_original = 0.0
    if views_original[index_comp].ratio() != 0:
        ratio_original = views_original[index].ratio() / views_original[index_comp].ratio()
    ratio_target = 0.0
    if views[index_comp].ratio() != 0:
        ratio_target = views[index].ratio() / views[index_comp].ratio()
    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(ratio_original)
        features.append(ratio_target)
    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        features.append(ratio_original - ratio_target)
    return features
