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


# other possible features: angle, symmetry score

# 11 features
def compute_horizontal_distance_one_screen(views, index, index_comp, mode):
    features = []
    distll = views[index].x - views[index_comp].x
    distlr = views[index].x - views[index_comp].x_end()
    distrr = views[index].x_end() - views[index_comp].x_end()
    distrl = views[index].x_end() - views[index_comp].x

    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(distll)
        features.append(distlr)
        features.append(distrr)
        features.append(distrl)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        # alignment
        features.append(distll == 0)
        features.append(distlr == 0)
        features.append(distrl == 0)
        features.append(distrr == 0)

        # centering, view is centered in view comp
        features.append(distll == -distrr)

        # overlap
        features.append(views[index].x >= views[index_comp].x)
        features.append(views[index].x <= (views[index_comp].x + views[index_comp].width))

    return features


# 11 features
def compute_vertical_distance_one_screen(views, index, index_comp, mode):
    features = []
    disttt = views[index].y - views[index_comp].y
    disttb = views[index].y - views[index_comp].y_end()
    distbb = views[index].y_end() - views[index_comp].y_end()
    distbt = views[index].y_end() - views[index_comp].y

    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(disttt)
        features.append(disttb)
        features.append(distbb)
        features.append(distbt)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        features.append(disttt == 0)
        features.append(disttb == 0)
        features.append(distbb == 0)
        features.append(distbt == 0)

        # centering
        features.append(disttt == -distbb)

        # overlap
        features.append(views[index].y >= views[index_comp].y)
        features.append(views[index].y <= (views[index_comp].y + views[index_comp].height))

    return features


# 4 features
def compute_dimension_one_screen(views, index, index_comp, mode):
    features = []
    h_diff = views[index].height - views[index_comp].height
    w_diff = views[index].width - views[index_comp].width

    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(h_diff)
        features.append(w_diff)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        # same dimensions
        features.append(h_diff == 0)
        features.append(w_diff == 0)

    return features


# 2 features
def compute_ratio_one_screen(views, index, index_comp, mode):
    features = []
    ratio_target = 0.0
    if views[index_comp].ratio() != 0:
        ratio_target = views[index].ratio() / views[index_comp].ratio()
    if mode == DataModeRNN.FULL or mode == DataModeRNN.RAW:
        features.append(ratio_target)

    if mode == DataModeRNN.FULL or mode == DataModeRNN.ABSTRACT:
        # same ratio
        features.append(ratio_target == 1)
    return features
