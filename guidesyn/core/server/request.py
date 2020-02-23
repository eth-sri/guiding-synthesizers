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

import requests

# supported models: {"MLP", "doubleRNN", "CNN", "EnsembleRnnCnn"};
req = {'model': 'RNN', 'dataset': 'ds', 'devices': [{'dimensions': [1400, 2520], 'filename': 'filename', 'layouts': [{'Views': [{'height': 2520, 'id': 0, 'width': 1400, 'x': 0, 'y': 0}, {'height': 1424, 'id': 1, 'width': 1440, 'x': -40, 'y': 500}], 'layout_id': -1}], 'original': {'Views': [{'height': 2560, 'id': 0, 'width': 1440, 'x': 0, 'y': 0}, {'height': 1424, 'id': 1, 'width': 1440, 'x': 0, 'y': 520}, {'height': 1264, 'id': 2, 'width': 1280, 'x': 80, 'y': 600}, {'height': 688, 'id': 3, 'width': 1152, 'x': 144, 'y': 856}, {'height': 192, 'id': 4, 'width': 1152, 'x': 144, 'y': 1608}, {'height': 112, 'id': 5, 'width': 1152, 'x': 144, 'y': 664}], 'layout_id': -1}, 'target': {'Views': [{'height': 2520, 'id': 0, 'width': 1400, 'x': 0, 'y': 0}, {'height': 0, 'id': 1, 'width': 0, 'x': -1, 'y': -1}, {'height': 0, 'id': 2, 'width': 0, 'x': -1, 'y': -1}, {'height': 0, 'id': 3, 'width': 0, 'x': -1, 'y': -1}, {'height': 0, 'id': 4, 'width': 0, 'x': -1, 'y': -1}, {'height': 0, 'id': 5, 'width': 0, 'x': -1, 'y': -1}], 'layout_id': -1}}], 'generateData': False}

print("Request for http://localhost:4446/predict")
res = requests.post('http://localhost:4446/predict', json=req)
print(res.json()["scores"])

