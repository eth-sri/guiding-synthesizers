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


def draw_rectangle(draw, x, y, width, height):
    width_ = width - 1
    height_ = height - 1
    draw.line((x, y, x + width_, y), fill=0)
    draw.line((x, y + height_, x + width_, y + height_), fill=0)
    draw.line((x + width_, y, x + width_, y + height_), fill=0)
    draw.line((x, y, x, y + height_), fill=0)


class Constraint:
    def __init__(self, _type, prob, size, val_primary, val_secondary, bias, srcid, tgt_prim, tgt_scnd):
        self.type = _type
        self.prob = prob
        self.size = size
        self.val_primary = val_primary
        self.val_secondary = val_secondary
        self.bias = bias
        self.srcid = srcid
        self.tgt_prim = tgt_prim
        self.tgt_scnd = tgt_scnd

    @classmethod
    def constraint_from_json(cls, json):
        return cls(json['type'], float(json['prob']), json['size'], int(json['val_primary']),
                   int(json['val_secondary']), float(json['bias']), int(json['srcid']), int(json['tgt_prim']),
                   int(json['tgt_scnd']))

    def to_string(self):
        return self.type + ", " + str(self.prob) + ", " + self.size + ", " + str(self.val_primary) + ", " + str(
            self.val_secondary) + ", " + str(self.bias) + ", " + str(self.srcid) + ", " + str(
            self.tgt_prim) + ", " + str(self.tgt_scnd)

    def equal(self, constraint):
        return self.type == constraint.type and self.size == constraint.size and self.val_primary == constraint.val_primary and self.val_secondary == constraint.val_secondary and self.bias == constraint.bias and self.srcid == constraint.srcid and self.tgt_prim == constraint.tgt_prim and self.tgt_scnd == constraint.tgt_scnd

    def equal_type(self, constraint):
        return self.type == constraint.type


class View:
    counter = 0

    def __init__(self, x, y, width, height, prob_horizontal=None, prob_vertical=None):
        self.id = View.counter
        View.counter = View.counter + 1
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.has_prob_info = False
        self.has_constraint_info = False
        # helper attribute for sorting views -> do not rely one this
        self.sortId = -1
        if not (prob_horizontal is None or prob_vertical is None):
            self.prob_horizontal = prob_horizontal
            self.prob_vertical = prob_vertical
            self.has_prob_info = True

    def add_constraints(self, vert_constraint, hori_constraint):
        self.has_constraint_info = True
        self.vertical_constraint = vert_constraint  # Constraint.constraintFromJSON(json_vert)
        self.horizontal_constraint = hori_constraint  # Constraint.constraintFromJSON(json_hor)

    def value_in_range(self, value, _min, _max):
        return (value > _min) and (value < _max)

    def value_in_range_including(self, value, _min, _max):
        return (value >= _min) and (value <= _max)

    def intersect(self, view):
        if self.contains(view) or view.contains(self):
            return False
        x_overlap = self.value_in_range(self.x, view.x, view.x + view.width) or self.value_in_range(view.x, self.x,
                                                                                                    self.x + self.width)
        y_overlap = self.value_in_range(self.y, view.y, view.y + view.height) or self.value_in_range(view.y, self.y,
                                                                                                     self.y + self.height)
        return x_overlap and y_overlap

    def intersect_including(self, view):
        return self.x_overlap(view) and self.y_overlap(view)

    def inside_screen(self, maximum_view_width, maximum_view_height):
        return self.x >= 0 and self.y >= 0 and (self.x + self.width) <= maximum_view_width and (
                self.y + self.height) <= maximum_view_height

    def is_centered_horizontally(self, maximum_view_width):
        return self.x == (maximum_view_width - (self.width + self.x))

    def area(self):
        return self.width * self.height

    def equal(self, other_view):
        return int(self.x) == int(other_view.x) and int(self.y == other_view.y) and int(
            self.width == other_view.width) and int(self.height) == int(other_view.height)

    def x_end(self):
        return self.x + self.width

    def y_end(self):
        return self.y + self.height

    def ratio(self):
        if float(self.height) == 0.0:
            # print("Height is 0. Weird ratio")
            return 0.0
        return float(self.width) / float(self.height)

    def same_dimensions(self, view):
        return int(self.width) == int(view.width) and int(self.height) == int(view.height)

    # from data_reader view
    def draw_downsampled(self, draw, downsample, randoffsetx=0, randoffsety=0):
        x = int(self.x / downsample) + int(randoffsetx)
        y = int(self.y / downsample) + int(randoffsety)
        width = int(self.width / downsample)
        height = int(self.height / downsample)
        draw_rectangle(draw, x, y, width, height)

    # from data_reader view
    def file_string(self):
        string = str(self.x) + ", " + str(self.y) + ", " + str(self.width) + ", " + str(self.height)
        if self.has_prob_info:
            string = string + ", " + str(self.prob_horizontal) + ", " + str(self.prob_vertical)
        if self.has_constraint_info:
            string = string + ", " + self.vertical_constraint.to_string() + ", " + self.horizontal_constraint.to_string()
        return string

    def file_string_formatted(self):
        string = str(self.x) + "," + str(self.y) + "," + str(self.x + self.width) + "," + str(self.y + self.height)
        return string

    def y_overlap(self, view):
        return self.value_in_range_including(self.y, view.y, view.y + view.height) or self.value_in_range_including(
            view.y,
            self.y,
            self.y + self.height)

    def x_overlap(self, view):
        return self.value_in_range_including(self.x, view.x, view.x + view.width) or self.value_in_range_including(
            view.x,
            self.x,
            self.x + self.width)

    def contains(self, view):
        return self.x <= view.x and self.x_end() >= view.x_end() and self.y <= view.y and self.y_end() >= view.y_end()

    def diff(self, view):
        return [self.x - view.x, self.y - view.y, self.width - view.width, self.height - view.height]

    def to_dict(self):
        return {"id": self.id, "x": self.x, "y": self.y, "width": self.width, "height": self.height, "prob_vert": 0,
                "prob_hori": 0}

    def to_array(self):
        # warning since synthesizer (C++) expects coordindates in the form (x, y, xend, yend)
        return [self.x, self.y, self.width + self.x, self.height + self.y]

    # transforms coordinates
    def _pix_to_dp(self, factor):
        self.x = int(self.x / factor)
        self.y = int(self.y / factor)
        self.width = int(self.width / factor)
        self.height = int(self.height / factor)
