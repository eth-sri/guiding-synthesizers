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

from __future__ import print_function
from flask import request
from flask import jsonify
import json
import torch
from PIL import Image, ImageDraw
import numpy as np
import os
from flask import Flask
from flask_restful import Api, Resource, reqparse
from guidesyn.core.server.server_helper import predict_example, resetModel, equal
from guidesyn.core.view import View, Constraint
from guidesyn.core.arguments import Data
import argparse

parser = argparse.ArgumentParser(description='Server specifications')
parser.add_argument('--model_path', type=str, default="./core/saved_modelsPaper/",
                    help='Path to load models from/save to')
args = parser.parse_args()

app = Flask(__name__)


def views_string(views):
    string = ""
    for _view in views:
        string += (_view.file_string_formatted() + "\n")
    return string


def drawViews(path_name, views, app_id, layout_append, downsample=1, device_id=-1):
    downsample = 4
    device_width = views[0].width / downsample
    device_height = views[0].height / downsample
    image = Image.new('RGB', (int(device_width), int(device_height)))
    draw = ImageDraw.Draw(image)
    draw.rectangle(((0, 0), (device_width + 1, device_height + 1)), fill="white")
    for view in views:
        view.draw_downsampled(draw, int(downsample))

    if device_id == -1:
        save_string = path_name + str(app_id) + "_" + str(len(views)) + "_" + layout_append + ".png"
    else:
        save_string = path_name + str(app_id) + "_d=" + str(device_id) + "_" + str(
            len(views)) + "_" + layout_append + ".png"

    file = open(save_string.replace(".png", ".txt"), "w")
    file.write(views_string(views))
    file.close()
    image.save(save_string, "PNG")


def visualize(candidate_app, content, filename, original_views, score, target_app, model, dataset):
    view_number = len(candidate_app)
    candidate_id = content["layouts"][0]["layout_id"]
    selectedCorrectCandidate = candidate_app[view_number - 1].equal(target_app[view_number - 1])
    path_name = "./visualisations/" + model + "/" + dataset + "/" + filename + "/"
    if not os.path.isdir(path_name):
        os.makedirs(path_name)
    assert (len(score) == 1)
    drawViews(path_name, candidate_app, filename,
              str(candidate_id) + "_" + str(round(score[0], 4)).replace(".", "") + "_" + str(
                  int(selectedCorrectCandidate)))
    drawViews(path_name, original_views, filename, "original")
    drawViews(path_name, target_app, filename, "target")


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        print("Invalid input! Input is not a json file.")
    all_devices = request.get_json()
    print(len(all_devices))
    print(all_devices["model"])
    dataset = all_devices["dataset"]

    score_sum = np.zeros(len(all_devices["devices"][0]["layouts"]))
    for device_id, content in enumerate(all_devices["devices"]):
        filename = content["filename"]
        dimensions = content["dimensions"]
        # device_width =  float(dimensions[0])
        # device_height = float(dimensions[1])
        (apps, views, result_index, score, downsample, targetScore, trueId, _) = predict_example(content,
                                                                                                 args.model_path,
                                                                                                 server_name=
                                                                                                 all_devices["model"],
                                                                                                 dataset=all_devices[
                                                                                                     "dataset"],
                                                                                                 targetXML=all_devices.get(
                                                                                                     "targetXML", None))
        score_sum = score_sum + np.asarray(score, dtype=np.float32)

        original_views = [] # different dimension
        for view_json in content["original"]["Views"]:
            original_view = View(int(view_json['x']), int(view_json['y']), int(view_json['width']),
                                 int(view_json['height']), 0.0, 0.0)
            # expect the layouts to be scored -> int(view_json['id'])-1 is the position in the array
            if ("targetXML" in all_devices) and (all_devices["targetXML"] is not None) and (int(view_json['id']) != 0):
                original_view.add_constraints(
                    Constraint.constraint_from_json(all_devices["targetXML"]["vertical"][int(view_json['id']) - 1]),
                    Constraint.constraint_from_json(all_devices["targetXML"]["horizontal"][int(view_json['id']) - 1]))
            original_views.append(original_view)
            if len(original_views) == len(views):
                break

        # visualize predicted results
        visualize(apps[0], content, filename, original_views, score, views, all_devices["model"], dataset)

        # file = open("./ServerScreens/compared_" + str(filename) + "_" + str(len(views)) + "_.txt","w")
        # file.write(''.join(str(e)+"\n" for e in score))
        # file.write("\nSelected: " + str(result_index) + "       " + str(score[result_index]))
        # file.write("\nTarget score: " + str(targetScore))
        # file.write("\nFound in: ")

        # To evaluate execution on only the first device!!!
        break

    max_score_indexes = []
    # for idx, val in enumerate(score):
    #    if val == targetScore:
    #        file.write(str(idx) + ", ")
    #        maxScoreIndexes.append(idx)
    # file.write("\nMatching in: ")
    # for idx, val in enumerate(apps):
    #    if equal(val, views): 
    #        file.write(str(idx) + ", ")
    # file.close()
    # if(not os.path.isdir("./Requests-" + dataset)):
    #     os.mkdir("./Requests-" + dataset)    
    # with open("./Requests-" + dataset + "/request_" + str(filename) + "_" + str(len(views)) + "_.txt","w") as outfile:
    #     json.dump(all_devices, outfile)

    selected_sum = np.argmax(score_sum).item()
    print(score_sum.tolist())
    # if trueId is not equal to selected_sum, it was not necessarily a wrong choice if this was not the distinguishing device
    # print("Selected sum: ", selected_sum, trueId)
    return jsonify(result=selected_sum, results=max_score_indexes,
                   scores=score_sum.tolist())  # return id depending on which one is the correct one


@app.route('/visualize', methods=['POST'])
def visualize_example():
    if not request.is_json:
        print("Invalid input! Input is not a json file.")
    content = request.get_json()
    model = content["model"]
    dataset = content["dataset"]
    filename = content["appId"]

    device_id = content["device_id"]
    correct_views = content["correct_views"]
    all_views = content["all_views"]
    path_name = "./visualisations/" + model + "/" + dataset + "/" + str(filename) + "/"
    generated_app = []
    for view_json in content["generated_app"]["Views"]:
        generated_view = View(int(view_json['x']), int(view_json['y']), int(view_json['width']),
                              int(view_json['height']), 0.0, 0.0)
        generated_app.append(generated_view)
    drawViews(path_name, generated_app, filename,
              "generated_" + str(device_id) + "_" + str(correct_views) + "_out_of_" + str(all_views))
    target_app = []
    for view_json in content["reference_app"]["Views"]:
        target_view = View(int(view_json['x']), int(view_json['y']), int(view_json['width']), int(view_json['height']),
                           0.0, 0.0)
        target_app.append(target_view)
    drawViews(path_name, target_app, filename,
              "target_" + str(device_id) + "_" + str(correct_views) + "_out_of_" + str(all_views))
    return jsonify(status="success")


app.run(debug=True, port=4446, use_reloader=False)
