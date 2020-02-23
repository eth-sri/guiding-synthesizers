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
from random import randint
import config


class View:
    def __init__(self, _id, x, y, width, height, hierarchy_id=-1, children_view_ids=[]):
        self.id = _id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.hierarchyId = hierarchy_id
        self.childrenViewIds = children_view_ids

    def draw_downsampled(self, draw):
        x = int(self.x / config.downsample)
        y = int(self.y / config.downsample)
        width = int(self.width / config.downsample)
        height = int(self.height / config.downsample)
        draw_rectangle(draw, x, y, width, height)

    def x_end(self):
        return self.x + self.width

    def y_end(self):
        return self.y + self.height

    def view_string_downsampled(self):
        return "View " + str(self.id) + ": " + str(self.x / config.downsample) + ", " + str(
            self.y / config.downsample) + ", " + str(self.width / config.downsample) + ", " + str(
            self.height / config.downsample) + "\n" + str(hierarchyId) + ", " + str(len(childrenViewIds))

    def view_string(self):
        return "View " + str(self.id) + ": " + str(self.x) + ", " + str(self.y) + ", " + str(self.width) + ", " + str(
            self.height) + "\n" + str(self.hierarchyId) + ", " + str(len(self.childrenViewIds))

    def view_string_end(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.x_end()) + ", " + str(self.y_end())

    def file_string(self):
        return str(self.x) + ", " + str(self.y) + ", " + str(self.width) + ", " + str(self.height)

    def equal_size(self, view):
        return self.x == view.x and self.y == view.y and self.width == view.width and self.height == view.height

    def print_downsampled(self):
        print(self.viewString_downsampled)

    def print_view(self):
        print("View " + str(self.id) + ": " + str(self.x) + ", " + str(self.y) + ", " + str(self.width) + ", " + str(
            self.height))

    def copy(self):
        return View(self.id, self.x, self.y, self.width, self.height, self.hierarchyId, self.childrenViewIds)

    def equal_coordinates(self, view):
        return self.x == view.x and self.y == view.y and self.width == view.width and self.height == view.height

    def area(self):
        return self.width * self.height
