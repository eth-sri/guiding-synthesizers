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

import config
from PIL import Image, ImageDraw
from random import randint, uniform


def draw_subviews(views, view, image_name):
    image = Image.new('RGB', (int(config.image_width), int(config.image_height)))
    draw = ImageDraw.Draw(image, 'RGBA')
    draw.rectangle(((0, 0), (config.image_width + 1, config.image_height + 1)), fill="white")
    view.draw_downsampled(draw)
    # view.print_view()
    for _id in view.childrenViewIds:
        subview = views[_id]
        # subview.print_view()
        subview.draw_downsampled(draw)
    image.save(image_name, "PNG")


def draw_views(views, image_name):
    image = Image.new('RGB', (int(config.image_width), int(config.image_height)))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 0), (config.image_width + 1, config.image_height + 1)), fill="white")
    for view in views:
        view.draw_downsampled(draw)
        # for visualisation, e.g. bad example
    image.save(image_name, "PNG")


def views_string(views, view=None):
    string = ""
    if view is None:
        for _view in views:
            string = string + (_view.file_string() + "\n")
    else:
        string = string + (view.file_string() + "\n")
        for _id in view.childrenViewIds:
            subview = views[_id]
            string = string + (subview.file_string() + "\n")
    return string
