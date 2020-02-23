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

# original screen
screen_width = 1440  # 360
screen_height = 2560  # 640

eval_rate = 15  # every 15th sample is for eval
number_of_candidates_upper = 16
number_of_candidates_lower = 3

number_of_view_modifications_upper = 4
modify_more_than_last_view = False  # also for positive example, attempt to make nets more robust to changes mistakes in earlier iterations
