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


# compute intersection
def compute_intersections(views):
    intersects = 0
    for i in range(0, len(views)):
        for j in range(i + 1, len(views)):
            view = views[j]
            intersects = intersects + views[i].intersect(view)
    return float(intersects)


def inside_screen(views, maximum_view_width, maximum_view_height):
    inside_screen_views = 0
    for view in views:
        inside_screen_views += view.inside_screen(maximum_view_width, maximum_view_height)
    return float(inside_screen_views)


def compute_similar_alignment_horizontally(views):
    aligned = 0
    for i in range(0, len(views)):
        for j in range(i + 1, len(views)):
            v1 = views[i]
            v2 = views[j]
            aligned = aligned + int(
                v1.x == v2.x or v1.x == v2.x_end() or v1.x_end() == v2.x or v1.x_end() == v2.x_end())
    return aligned


def compute_similar_alignment_vertically(views):
    aligned = 0
    for i in range(0, len(views)):
        for j in range(i + 1, len(views)):
            v1 = views[i]
            v2 = views[j]
            aligned = aligned + int(
                v1.y == v2.y or v1.y == v2.y_end() or v1.y_end() == v2.y or v1.y_end() == v2.y_end())
    return aligned


def compute_centered_horizontally(views):
    centered = 0
    for i, view in enumerate(views):
        for j, other_view in enumerate(views):
            # view.x is centered in other_view.x
            if i != j and (view.x - other_view.x) == (other_view.x_end() - view.x_end()):
                centered = centered + 1
    return float(centered)


def compute_centered_vertically(views):
    centered = 0
    for i, view in enumerate(views):
        for j, other_view in enumerate(views):
            # view.x is centered in other_view.x
            if i != j and (view.y - other_view.y) == (other_view.y_end() - view.y_end()):
                centered = centered + 1
    return float(centered)


def compute_centered_horizontally_different_views(views):
    centered = 0
    for i, v1 in enumerate(views):
        for j, v2 in enumerate(views):
            for k, v3 in enumerate(views):
                if i != j or j != k or i != k:
                    # v2 is centered between v3 and v1
                    if ((v2.x - v1.x_end() == v3.x - v2.x_end()) or (v2.x - v1.x == v3.x - v2.x_end()) or (
                            v2.x - v1.x_end() == v3.x_end() - v2.x_end()) or (v1.x - v2.x == v2.x_end() - v3.x_end())):
                        centered = centered + 1
    return float(centered)


def compute_centered_vertically_different_views(views):
    centered = 0
    for i, v1 in enumerate(views):
        for j, v2 in enumerate(views):
            for k, v3 in enumerate(views):
                if i != j or j != k or i != k:
                    # v2 is centered between v3 and v1, v2 is the centered one
                    if ((v2.y - v1.y_end() == v3.y - v2.y_end()) or (v2.y - v1.y == v3.y - v2.y_end()) or (
                            v2.y - v1.y_end() == v3.y_end() - v2.y_end()) or (v1.x - v2.x == v2.y_end() - v3.y_end())):
                        centered = centered + 1
    return float(centered)


def compute_same_dimensions_score(views):
    same_dim = 0
    for i, view in enumerate(views):
        for j, other_view in enumerate(views):
            if i != j and (view.width == other_view.width and view.height == other_view.height):
                same_dim = same_dim + 1
    return same_dim


# ratios = [1.0/1.0, 3.0/4.0, 4.0/3.0, 9.0/16.0, 9.0/16.0]
def popular_aspect_ratio(views, ratios):
    ratio_count = 0
    for idx, val in enumerate(views):
        if val.ratio() in ratios:
            ratio_count = ratio_count + 1
    return ratio_count


# [0, 8, 14, 16, 20, 24, 30, 32, 48]
# popular_margin_horizontal with 0 is the same case as alignment
def popular_margin_horizontal(views, margins):
    margin_count = 0
    for i in range(0, len(views)):
        for j in range(i + 1, len(views)):
            if abs(int(views[i].x - views[j].x)) in margins or abs(
                    int(views[i].x_end() - views[j].x)) in margins or abs(
                    int(views[i].x - views[j].x_end())) in margins or abs(
                int(views[i].x_end() - views[j].x_end())) in margins:
                margin_count = margin_count + 1
    return margin_count


def popular_margin_vertical(views, margins):
    margin_count = 0
    for i in range(0, len(views)):
        for j in range(i + 1, len(views)):
            if abs(int(views[i].y - views[j].y)) in margins or abs(
                    int(views[i].y_end() - views[j].y)) in margins or abs(
                    int(views[i].y - views[j].y_end())) in margins or abs(
                int(views[i].y_end() - views[j].y_end())) in margins:
                margin_count = margin_count + 1
    return margin_count


def add_raw_coordinates(views, array):
    for view in views:
        array.append(view.x)
        array.append(view.y)
        array.append(view.x + view.width)
        array.append(view.y + view.height)
